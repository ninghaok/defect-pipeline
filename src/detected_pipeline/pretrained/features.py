"""Image preparation and frozen feature networks for the ADPretrain + PatchCore detector."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "adpretrain"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from detected_pipeline.roi import read_image  # noqa: E402

FILL_BGR = np.asarray((104, 116, 123), dtype=np.uint8)   # ImageNet mean, outside-ROI fill (matches the validated experiment)


class RoiCropper:
    """Optional white-is-inspect ROI: crop to its bounding box, fill outside with the mean colour."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path).resolve() if path else None
        self.template = (read_image(self.path, cv2.IMREAD_GRAYSCALE) >= 128) if self.path else None
        if self.template is not None and not self.template.any():
            raise ValueError(f"ROI mask has no selected pixels: {self.path}")

    def mask_for_shape(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        if self.template is None:
            return np.ones((h, w), bool)
        mask = self.template
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if not mask.any():
            raise ValueError(f"ROI is empty after resizing: {self.path}")
        return mask

    def apply(self, image: np.ndarray):
        """Return (cropped image, valid mask of the crop, bbox x1,y1,x2,y2 in the full frame)."""
        mask = self.mask_for_shape(image.shape[:2])
        ys, xs = np.where(mask)
        x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        cropped = image[y1:y2, x1:x2].copy(); valid = mask[y1:y2, x1:x2]
        cropped[~valid] = FILL_BGR
        return cropped, valid, (x1, y1, x2, y2)


def letterbox(image: np.ndarray, size: int):
    h, w = image.shape[:2]; s = min(size / h, size / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top, left = (size - nh) // 2, (size - nw) // 2
    boxed = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left, cv2.BORDER_REFLECT_101)
    return boxed, (top, left, nh, nw)


def unbox(values: np.ndarray, shape: tuple[int, int], meta, nearest: bool = False) -> np.ndarray:
    top, left, nh, nw = meta
    return cv2.resize(values[top:top + nh, left:left + nw], (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


class Transform:
    def __init__(self):
        self.normalize = Normalize((.485, .456, .406), (.229, .224, .225))

    def __call__(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.normalize(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255))


class FeatureNetworks:
    """Frozen DINOv2 encoder + ADPretrain angle projector (loaded once, shared by every category)."""

    def __init__(self, dino_weight: Path, angle_weight: Path, backbone: str = "dinov2-large", device: str = "cuda:0"):
        from models.dino import DinoModel
        from models.projector import MultiScaleAttentionProjector
        self.device = device
        self.encoder = DinoModel(backbone, device=device, weight_path=str(dino_weight)).to(device).eval()
        self.projector = MultiScaleAttentionProjector(self.encoder.feature_dimensions, device=device)
        state = torch.load(angle_weight, map_location="cpu", weights_only=False)
        self.projector.load_state_dict(state["projectors"], strict=True)
        self.projector = self.projector.to(device).eval()
        self.transform = Transform()

    def tensor(self, image: np.ndarray, size: int):
        boxed, meta = letterbox(image, size)
        return self.transform(boxed).unsqueeze(0), meta

    @torch.inference_mode()
    def global_descriptor(self, batch: torch.Tensor) -> torch.Tensor:
        fmap = self.encoder.encode_image_from_tensors(batch.to(self.device), shape="img")[-1]
        return F.normalize(fmap.mean(dim=(2, 3)).float(), dim=1).cpu()

    @torch.inference_mode()
    def patch_features(self, batch: torch.Tensor):
        features = self.encoder.encode_image_from_tensors(batch.to(self.device), shape="img")
        return tuple(x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).contiguous() for x in features)
