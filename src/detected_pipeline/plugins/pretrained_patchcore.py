"""Pretrained (cold-start) detector plugin: ADPretrain residual features + PatchCore, hybrid segmentation.

Classification: image score = top 0.5% heat-map mean; NG when score >= image threshold.  The threshold is
recalibrated by the lifecycle as confirmed NG accumulate (see ``calibration.choose_image_threshold``).
Segmentation: only for NG images, ``hybrid_regions`` on the same heat map (absolute q99.7 of calibration OK
pixels vs 0.7 x image peak, whichever is higher), top-k regions ranked by peak.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detected_pipeline.calibration import choose_image_threshold, quantile_higher
from detected_pipeline.contracts import Decision, InferenceContext, PretrainedPrediction
from detected_pipeline.roi import read_image, write_image
from detected_pipeline.util import atomic_write_json, sha256_file, utc_now


class PretrainedPatchCorePlugin:
    plugin_name = "adpretrain_patchcore_224_hybrid"
    plugin_version = "3"

    def __init__(self):
        self.loaded = False
        self.config: dict[str, Any] = {}
        self.detector = None
        self.thresholds: dict[str, dict[str, Any]] = {}
        self.cal_ok: dict[str, list[Path]] = {}

    # ------------------------------------------------------------------ setup
    def load(self, config: dict[str, Any]) -> None:
        from detected_pipeline.pretrained import PretrainedDetector
        from detected_pipeline.pretrained.features import FeatureNetworks
        self.config = dict(config)
        root = Path(config["project_root"])
        self.cache_root = Path(config["cache_root"]); self.results_root = Path(config["results_root"])
        self.roi_masks = {k: (root / v) for k, v in dict(config.get("roi_masks", {})).items()}
        self.rules = dict(config.get("thresholds", {}))
        self.segmentation = dict(config.get("segmentation", {}))
        self.boundary_ratio = float(config.get("boundary_ratio", 0.05))
        networks = FeatureNetworks(root / config["dino_weight"], root / config["angle_weight"],
                                   config.get("backbone", "dinov2-large"), config.get("device", "cuda:0"))
        self.detector = PretrainedDetector(networks, config)
        self.model_version = "adpretrain224-" + sha256_file(root / config["angle_weight"])[:12]
        self.loaded = True

    def prepare_category(self, category: str, train_ok: list[Path], calibration_ok: list[Path]) -> dict[str, Any]:
        """Build the memory bank, score the calibration OK images and set the zero-NG thresholds."""
        cache = self.cache_root / category
        state = self.detector.build(category, train_ok, self.roi_masks.get(category), cache)
        self.cal_ok[category] = list(calibration_ok)
        scores_path = cache / "calibration_ok_scores.json"
        expected = [sha256_file(p) for p in calibration_ok]
        cached = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else None
        if not cached or cached.get("sha256") != expected or cached.get("fingerprint") != state["manifest"]["fingerprint"]:
            scores = [self.detector.score_image(category, p) for p in calibration_ok]
            pixels = self.detector.ok_pixel_sample(category, calibration_ok)
            cached = {"category": category, "fingerprint": state["manifest"]["fingerprint"], "sha256": expected,
                      "images": [str(p) for p in calibration_ok], "scores": scores,
                      "pixel_threshold": quantile_higher(pixels, float(self.segmentation.get("pixel_quantile", 0.997)))}
            atomic_write_json(scores_path, cached)
        state["ok_scores"] = [float(s) for s in cached["scores"]]
        state["pixel_threshold"] = float(cached["pixel_threshold"])
        threshold_path = cache / "thresholds.json"
        if threshold_path.exists():
            self.thresholds[category] = json.loads(threshold_path.read_text(encoding="utf-8"))
        else:
            self.calibrate(category, [])
        return state

    def calibrate(self, category: str, confirmed_ng: list[Path]) -> dict[str, Any]:
        """(Re)select the image threshold from calibration OK scores and the confirmed NG scores."""
        state = self.detector.categories[category]
        ng_scores = [self.detector.score_image(category, p) for p in confirmed_ng]
        result = choose_image_threshold(state["ok_scores"], ng_scores, self.rules, category)
        result.update({"image_threshold": result["threshold"], "pixel_threshold": state["pixel_threshold"],
                       "ng_images": [str(p) for p in confirmed_ng], "ng_scores": ng_scores, "calibrated_at": utc_now()})
        previous = self.thresholds.get(category, {})
        result["history"] = (previous.get("history") or []) + [{k: previous[k] for k in ("rule", "image_threshold", "calibration_ng", "calibrated_at") if k in previous}] if previous else []
        self.thresholds[category] = result
        atomic_write_json(self.cache_root / category / "thresholds.json", result)
        return result

    # ------------------------------------------------------------------ inference
    def score(self, category: str, image_path: Path) -> float:
        return self.detector.score_image(category, image_path)

    def predict(self, image_path: Path, context: InferenceContext) -> PretrainedPrediction:
        from detected_pipeline.pretrained import hybrid_regions
        if not self.loaded or context.category not in self.thresholds:
            raise RuntimeError(f"category not prepared: {context.category}")
        thresholds = self.thresholds[context.category]
        started = time.perf_counter()
        image = read_image(image_path)
        heat, valid = self.detector.heatmap(context.category, image)
        score = self.detector.image_score(heat, valid)
        image_threshold = float(thresholds["image_threshold"])
        is_ng = score >= image_threshold
        stats: dict[str, Any] = {"regions": []}
        mask = np.zeros(heat.shape, bool)
        if is_ng:
            mask, stats = hybrid_regions(heat, valid, float(thresholds["pixel_threshold"]),
                                         float(self.segmentation.get("peak_fraction", 0.7)),
                                         int(self.segmentation.get("min_area", 128)), int(self.segmentation.get("max_regions", 3)))
        latency = (time.perf_counter() - started) * 1000.0
        output = self.results_root / context.category / context.sample_id
        output.mkdir(parents=True, exist_ok=True)
        heat_path, mask_path = output / "pixel_score.npy", output / "binary_mask.png"
        np.save(heat_path, heat)
        write_image(mask_path, mask.astype(np.uint8) * 255)
        atomic_write_json(output / "regions.json", {"image_score": score, "image_threshold": image_threshold, "decision": "NG" if is_ng else "OK", **stats})
        if is_ng:
            self._save_visual(image, heat, valid, mask, stats, output / "overlay.png")
        boundary = abs(score - image_threshold) <= abs(image_threshold) * self.boundary_ratio
        decision = Decision.NG if is_ng else Decision.OK
        return PretrainedPrediction(
            sample_id=context.sample_id, category=context.category,
            feature_norm_score=score, feature_norm_decision=decision,
            patchcore_score=score, patchcore_decision=decision, final_decision=decision,
            is_boundary=bool(boundary), is_conflict=False,
            pixel_score_map_path=str(heat_path), binary_mask_path=str(mask_path),
            thresholds_used={"image_threshold": image_threshold, "pixel_threshold": float(thresholds["pixel_threshold"]),
                             "peak_fraction": float(self.segmentation.get("peak_fraction", 0.7)),
                             "top_fraction": self.detector.top_fraction, "calibration_ng": float(thresholds.get("calibration_ng", 0))},
            latency_ms=latency, plugin_name=self.plugin_name, plugin_version=self.plugin_version, model_version=self.model_version,
        )

    @staticmethod
    def _save_visual(image, heat, valid, mask, stats, path: Path) -> None:
        values = heat[valid]; low, high = float(values.min()), float(values.max())
        normalized = np.zeros(heat.shape, np.uint8)
        if high > low:
            normalized[valid] = np.round((values - low) / (high - low) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        for region in stats.get("regions", []):
            x1, y1, x2, y2 = region["bbox_xyxy"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, f"#{region['rank']} {region['peak']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        write_image(path, overlay)

    def healthcheck(self) -> dict[str, Any]:
        return {"ok": self.loaded and self.detector is not None, "plugin": self.plugin_name, "model_version": getattr(self, "model_version", None)}

    def close(self) -> None:
        self.detector = None; self.loaded = False
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


def create_plugin() -> PretrainedPatchCorePlugin:
    return PretrainedPatchCorePlugin()
