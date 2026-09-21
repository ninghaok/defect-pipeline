"""ADPretrain residual features + PatchCore memory bank, single 224 view (the validated cold-start setting).

Per category:
  train OK  -> 32 k-center reference images (residual references) + PatchCore memory bank (all other OK, coreset)
  heat map  -> nearest-neighbour distance of every projected residual patch to the memory bank, upsampled
  score     -> mean of the top ``top_fraction`` heat-map pixels inside the ROI
The memory bank is cached on disk so a restart only re-encodes the 32 reference images.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from detected_pipeline.roi import read_image
from detected_pipeline.util import atomic_write_json, sha256_file

from .features import FeatureNetworks, RoiCropper, letterbox, unbox


def init_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    count = max(1, int(np.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean(dtype=np.float64))


class _SupportDataset(Dataset):
    def __init__(self, paths, size, transform, cropper):
        self.paths, self.size, self.transform, self.cropper = list(paths), size, transform, cropper

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image, _, _ = self.cropper.apply(read_image(self.paths[index]))
        boxed, _ = letterbox(image, self.size)
        return self.transform(boxed), 0, torch.zeros((1, self.size, self.size)), "normal"


class PretrainedDetector:
    def __init__(self, networks: FeatureNetworks, settings: dict[str, Any]):
        self.networks = networks
        self.size = int(settings.get("image_size", 224))
        self.reference_size = int(settings.get("reference_size", 32))
        self.coreset = float(settings.get("coreset_percentage", 0.02))
        self.top_fraction = float(settings.get("top_fraction", 0.005))
        self.seed = int(settings.get("seed", 42))
        self.batch_size = int(settings.get("batch_size", 4))
        self.kcenter_batch_size = int(settings.get("kcenter_batch_size", 16))
        self.device = networks.device
        self.categories: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ building
    def select_kcenter(self, paths: list[Path], count: int, cropper: RoiCropper) -> tuple[list[Path], list[dict]]:
        if len(paths) < count:
            raise ValueError(f"k-center needs {count} images, only {len(paths)} available")
        vectors = []
        for start in range(0, len(paths), self.kcenter_batch_size):
            chunk = paths[start:start + self.kcenter_batch_size]
            batch = torch.cat([self.networks.tensor(cropper.apply(read_image(p))[0], self.size)[0] for p in chunk])
            vectors.append(self.networks.global_descriptor(batch))
        features = torch.cat(vectors, 0)
        centroid = torch.nn.functional.normalize(features.mean(0, keepdim=True), dim=1)
        first = int((features @ centroid.T).squeeze(1).argmax())
        selected = [first]; minimum = 1.0 - (features @ features[first])
        steps = [{"rank": 1, "image": str(paths[first]), "distance": 0.0}]
        while len(selected) < count:
            minimum[selected] = -1.0
            index = int(minimum.argmax()); steps.append({"rank": len(selected) + 1, "image": str(paths[index]), "distance": float(minimum[index])})
            selected.append(index); minimum = torch.minimum(minimum, 1.0 - (features @ features[index]))
        return sorted((paths[i] for i in selected), key=str), steps

    def reference_bank(self, paths: list[Path], cropper: RoiCropper):
        out = [[], [], [], []]
        for path in paths:
            tensor, _ = self.networks.tensor(cropper.apply(read_image(path))[0], self.size)
            for destination, feature in zip(out, self.networks.patch_features(tensor)):
                destination.append(feature.detach())
        return tuple(torch.cat(x) for x in out)

    def _new_patchcore(self):
        from ad_models.patchcore import get_patchcore
        return get_patchcore(self.networks.encoder, self.networks.projector, device=self.device, residual=True,
                             input_shape=(3, self.size, self.size), nn_backend="torch", nn_query_chunk_size=2048,
                             coreset_percentage=self.coreset)

    def build(self, category: str, train_ok: list[Path], roi_path: Path | None, cache_dir: Path) -> dict[str, Any]:
        """Build (or restore from ``cache_dir``) the per-category reference bank and memory bank."""
        cropper = RoiCropper(roi_path)
        train_ok = sorted({sha256_file(p): p for p in train_ok}.values(), key=str)
        identity = {"images": [sha256_file(p) for p in train_ok], "roi": sha256_file(roi_path) if roi_path else None,
                    "image_size": self.size, "reference_size": self.reference_size, "coreset": self.coreset, "seed": self.seed}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path, bank_path = cache_dir / "bank_manifest.json", cache_dir / "memory_bank.npy"
        started = time.perf_counter()
        init_seeds(self.seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if manifest and manifest.get("fingerprint") == fingerprint and bank_path.exists():
            references = [Path(p) for p in manifest["references"]]
            bank = self.reference_bank(references, cropper)
            patchcore = self._new_patchcore()
            patchcore.anomaly_scorer.fit(detection_features=[np.load(bank_path)])
            restored = True
        else:
            references, steps = self.select_kcenter(train_ok, self.reference_size, cropper)
            support = [p for p in train_ok if p not in set(references)]
            bank = self.reference_bank(references, cropper)
            patchcore = self._new_patchcore()
            loader = DataLoader(_SupportDataset(support, self.size, self.networks.transform, cropper),
                                batch_size=self.batch_size, shuffle=False, num_workers=0)
            patchcore.fit(loader, bank, category, aligned=False)
            features = np.asarray(patchcore.anomaly_scorer.detection_features, dtype=np.float32)
            np.save(bank_path, features)
            manifest = {"category": category, "fingerprint": fingerprint, "references": [str(p) for p in references],
                        "kcenter_steps": steps, "support_images": len(support), "memory_bank_patches": int(features.shape[0]),
                        "settings": identity}
            atomic_write_json(manifest_path, manifest)
            restored = False
        state = {"cropper": cropper, "references": references, "bank": bank, "patchcore": patchcore,
                 "manifest": manifest, "setup_seconds": time.perf_counter() - started, "restored": restored}
        self.categories[category] = state
        return state

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def heatmap(self, category: str, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full-frame heat map and valid (ROI) mask for one BGR image."""
        state = self.categories[category]
        cropped, valid_crop, (x1, y1, x2, y2) = state["cropper"].apply(image)
        tensor, meta = self.networks.tensor(cropped, self.size)
        _, maps = state["patchcore"]._predict(tensor, state["bank"])
        crop_heat = unbox(np.asarray(maps[0], np.float32), cropped.shape[:2], meta)
        floor = float(crop_heat[valid_crop].min())
        crop_heat[~valid_crop] = floor
        heat = np.full(image.shape[:2], floor, np.float32); heat[y1:y2, x1:x2] = crop_heat
        valid = np.zeros(image.shape[:2], bool); valid[y1:y2, x1:x2] = valid_crop
        return heat, valid

    def image_score(self, heat: np.ndarray, valid: np.ndarray) -> float:
        return top_fraction_mean(heat[valid], self.top_fraction)

    def score_image(self, category: str, path: Path) -> float:
        heat, valid = self.heatmap(category, read_image(path))
        return self.image_score(heat, valid)

    def ok_pixel_sample(self, category: str, paths: list[Path], per_image: int = 20000) -> np.ndarray:
        rng = np.random.default_rng(self.seed); samples = []
        for path in paths:
            heat, valid = self.heatmap(category, read_image(path)); values = heat[valid]
            samples.append(values if values.size <= per_image else rng.choice(values, per_image, replace=False))
        return np.concatenate(samples)
