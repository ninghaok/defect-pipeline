"""YOLO instance-segmentation detector (detect first, then segment).

Image score = highest instance confidence above a very low ``conf_floor`` (0 when nothing is detected);
NG when score >= calibrated image threshold.  Masks are produced only for NG images: the union of the
instances whose confidence >= the calibrated ``mask_conf_threshold``.  For ROI categories the outside is
filled white before inference and instances that do not touch the ROI are dropped (same as in training).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detected_pipeline.contracts import Decision, InferenceContext, PretrainedPrediction, SupervisedPrediction
from detected_pipeline.roi import read_image, write_image
from detected_pipeline.util import atomic_write_json, sha256_file


def roi_for(path: Path | None, shape: tuple[int, int], cache: dict) -> np.ndarray | None:
    if path is None:
        return None
    key = (str(path), shape)
    if key not in cache:
        raw = read_image(path, cv2.IMREAD_GRAYSCALE)
        if raw.shape != shape:
            raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        cache[key] = raw >= 128
    return cache[key]


def prepare_image(image: np.ndarray, roi: np.ndarray | None) -> np.ndarray:
    if roi is None:
        return image
    out = image.copy(); out[~roi] = 255
    return out


class YoloSegDetector:
    """Raw detector shared by the plugin, calibration and the ablation experiment."""

    def __init__(self, checkpoint: Path, settings: dict[str, Any], roi_mask: Path | None = None):
        from ultralytics import YOLO
        self.checkpoint = Path(checkpoint)
        self.model = YOLO(str(self.checkpoint))
        self.imgsz = int(settings.get("imgsz", 1024)); self.conf_floor = float(settings.get("conf_floor", 0.001))
        self.nms_iou = float(settings.get("nms_iou", 0.7)); self.max_det = int(settings.get("max_det", 100))
        self.device = settings.get("device", 0)
        self.roi_mask = Path(roi_mask) if roi_mask else None
        self._roi_cache: dict = {}

    def infer(self, image: np.ndarray):
        """Return (score, confs[N], boxes[N,4] xyxy, masks[N,H,W] bool) on the full frame."""
        roi = roi_for(self.roi_mask, image.shape[:2], self._roi_cache)
        source = prepare_image(image, roi)
        result = self.model.predict(source=source, imgsz=self.imgsz, conf=self.conf_floor, iou=self.nms_iou,
                                    max_det=self.max_det, retina_masks=True, device=self.device, verbose=False)[0]
        n = 0 if result.boxes is None else len(result.boxes)
        if n and result.masks is not None:
            confs = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            masks = result.masks.data.detach().cpu().numpy() > 0.5
            if masks.shape[1:] != image.shape[:2]:
                masks = np.stack([cv2.resize(m.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
            if roi is not None:
                keep = np.array([(m & roi).any() for m in masks], bool)
                confs, boxes, masks = confs[keep], boxes[keep], masks[keep] & roi
        else:
            confs = np.zeros(0); boxes = np.zeros((0, 4)); masks = np.zeros((0,) + image.shape[:2], bool)
        return (float(confs.max()) if len(confs) else 0.0), confs, boxes, masks

    def score(self, image_path: Path) -> float:
        return self.infer(read_image(image_path))[0]

    @staticmethod
    def union(masks: np.ndarray, keep: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        return masks[keep].any(axis=0) if keep.any() else np.zeros(shape, bool)


class YoloSegPlugin:
    def __init__(self):
        self.detector: YoloSegDetector | None = None
        self.config: dict[str, Any] = {}

    def load(self, checkpoint: Path, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.detector = YoloSegDetector(checkpoint, config, config.get("roi_mask"))
        self.checkpoint_sha256 = sha256_file(Path(checkpoint))

    def predict(self, image_path: Path, context: InferenceContext) -> SupervisedPrediction:
        if self.detector is None:
            raise RuntimeError("YOLO plugin is not loaded")
        started = time.perf_counter()
        image = read_image(image_path)
        score, confs, boxes, masks = self.detector.infer(image)
        image_threshold = float(self.config["image_threshold"]); mask_threshold = float(self.config["mask_conf_threshold"])
        is_ng = score >= image_threshold
        keep = (confs >= mask_threshold) if is_ng else np.zeros(len(confs), bool)
        mask = self.detector.union(masks, keep, image.shape[:2])
        confidence_map = np.zeros(image.shape[:2], np.float32)
        for m, c in zip(masks, confs):
            confidence_map[m] = np.maximum(confidence_map[m], c)
        latency = (time.perf_counter() - started) * 1000.0
        output = Path(self.config["output_dir"]) / context.category / context.sample_id
        output.mkdir(parents=True, exist_ok=True)
        map_path, mask_path = output / "pixel_score.npy", output / "binary_mask.png"
        np.save(map_path, confidence_map)
        write_image(mask_path, mask.astype(np.uint8) * 255)
        instances = [{"rank": i + 1, "confidence": float(c), "bbox_xyxy": [float(v) for v in b], "kept": bool(k)}
                     for i, (c, b, k) in enumerate(sorted(zip(confs, boxes, keep), key=lambda z: -z[0]))]
        atomic_write_json(output / "regions.json", {"image_score": score, "image_threshold": image_threshold,
                                                     "mask_conf_threshold": mask_threshold, "decision": "NG" if is_ng else "OK",
                                                     "instances": instances[:20]})
        if is_ng:
            boxed = image.copy()
            for (x1, y1, x2, y2), c in zip(boxes[keep], confs[keep]):
                cv2.rectangle(boxed, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(boxed, f"{c:.2f}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            write_image(output / "boxed.jpg", boxed)
        return SupervisedPrediction(
            sample_id=context.sample_id, category=context.category, image_score=score,
            image_decision=Decision.NG if is_ng else Decision.OK,
            pixel_probability_path=str(map_path), binary_mask_path=str(mask_path),
            image_threshold=image_threshold, pixel_threshold=mask_threshold, latency_ms=latency,
            checkpoint_sha256=self.checkpoint_sha256, model_version=str(self.config["model_version"]),
        )

    def close(self) -> None:
        self.detector = None


class YoloFeedbackAdapter:
    """Expose a production YOLO-seg model through the feedback pipeline's normalized prediction contract."""

    def __init__(self, checkpoint: Path, config: dict[str, Any]):
        self.plugin = YoloSegPlugin(); self.checkpoint = Path(checkpoint); self.config = config

    def load(self, ignored: dict[str, Any] | None = None) -> None:
        self.plugin.load(self.checkpoint, self.config)

    def healthcheck(self) -> dict[str, Any]:
        return {"ok": self.plugin.detector is not None, "kind": "supervised"}

    def score(self, category: str, image_path: Path) -> float:
        return self.plugin.detector.score(image_path)

    def predict(self, image_path: Path, context: InferenceContext) -> PretrainedPrediction:
        result = self.plugin.predict(image_path, context)
        boundary = abs(result.image_score - result.image_threshold) <= result.image_threshold * float(self.config.get("boundary_ratio", 0.05))
        return PretrainedPrediction(
            sample_id=result.sample_id, category=result.category,
            feature_norm_score=result.image_score, feature_norm_decision=result.image_decision,
            patchcore_score=result.image_score, patchcore_decision=result.image_decision,
            final_decision=result.image_decision, is_boundary=bool(boundary), is_conflict=False,
            pixel_score_map_path=result.pixel_probability_path, binary_mask_path=result.binary_mask_path,
            thresholds_used={"image_threshold": result.image_threshold, "mask_conf_threshold": result.pixel_threshold,
                             "conf_floor": float(self.config.get("conf_floor", 0.001))},
            latency_ms=result.latency_ms, plugin_name="yolo26s-seg-supervised", plugin_version="3",
            model_version=result.model_version,
        )

    def close(self) -> None:
        self.plugin.close()
