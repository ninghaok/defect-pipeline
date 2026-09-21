from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RoiResult:
    image: np.ndarray
    roi_mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]


class WhiteMaskRoiCropper:
    """Crop to the white region of a fixed ROI mask.

    The ROI mask uses 255=inspect and 0=ignore.  Images are filled white
    outside the selected ROI before the bounding-box crop.  External ground
    truth masks use the dataset convention 255=background and 0=defect, so
    ignored pixels are forced to 255 before applying the identical crop.
    """

    def __init__(self, mask_path: str | Path, threshold: int = 128):
        self.mask_path = Path(mask_path).resolve()
        raw = cv2.imdecode(np.fromfile(self.mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError(f"cannot read ROI mask: {self.mask_path}")
        selected = raw >= int(threshold)
        if not selected.any():
            raise ValueError(f"ROI mask has no white selected pixels: {self.mask_path}")
        self._native = selected.astype(np.uint8)

    def mask_for_shape(self, height: int, width: int) -> np.ndarray:
        if (height, width) == self._native.shape:
            return self._native.astype(bool)
        resized = cv2.resize(self._native, (width, height), interpolation=cv2.INTER_NEAREST)
        selected = resized > 0
        if not selected.any():
            raise ValueError(f"resized ROI is empty for shape {(height, width)}")
        return selected

    @staticmethod
    def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def crop_image(self, image: np.ndarray) -> RoiResult:
        if image is None or image.ndim not in (2, 3):
            raise ValueError("invalid image for ROI crop")
        height, width = image.shape[:2]
        roi = self.mask_for_shape(height, width)
        prepared = image.copy()
        prepared[~roi] = 255
        x1, y1, x2, y2 = self.bbox(roi)
        return RoiResult(prepared[y1:y2, x1:x2], roi[y1:y2, x1:x2], (x1, y1, x2, y2))

    def crop_external_gt(self, mask: np.ndarray) -> RoiResult:
        if mask is None or mask.ndim != 2:
            raise ValueError("external ground-truth mask must be grayscale")
        height, width = mask.shape
        roi = self.mask_for_shape(height, width)
        prepared = mask.copy()
        prepared[~roi] = 255  # external convention: white is background
        x1, y1, x2, y2 = self.bbox(roi)
        return RoiResult(prepared[y1:y2, x1:x2], roi[y1:y2, x1:x2], (x1, y1, x2, y2))

    def cropped_mask_for_shape(self, height: int, width: int) -> RoiResult:
        roi = self.mask_for_shape(height, width)
        x1, y1, x2, y2 = self.bbox(roi)
        return RoiResult((roi[y1:y2, x1:x2].astype(np.uint8) * 255), roi[y1:y2, x1:x2], (x1, y1, x2, y2))


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path = Path(path)
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if value is None:
        raise ValueError(f"cannot read image: {path}")
    return value


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} else ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"cannot encode ROI output: {path}")
    encoded.tofile(path)
