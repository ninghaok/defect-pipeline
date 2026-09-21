"""Region extraction from a heat map for images already classified NG (hybrid rule).

Pixel threshold = max(absolute threshold from calibration OK pixels, peak_fraction x this image's peak).
The absolute term stops regions from flooding flat heat maps; the relative term tightens strong
responses to the defect itself.  Validated on dataset_523: IoU 0.14 -> 0.29 (oblique) and
0.06 -> 0.26 (bottom) with region area ~ defect area, hit rate unchanged.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def hybrid_regions(heat: np.ndarray, valid: np.ndarray, pixel_threshold: float, peak_fraction: float = 0.7,
                   min_area: int = 128, max_regions: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (binary mask = union of kept regions, stats with the ranked region list)."""
    if heat.shape != valid.shape:
        raise ValueError(f"heat/valid shape mismatch: {heat.shape} vs {valid.shape}")
    if not valid.any():
        raise ValueError("empty valid region")
    peak = float(heat[valid].max())
    threshold = max(float(pixel_threshold), peak_fraction * peak)
    binary = (heat >= threshold) & valid
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    regions = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        x, y, w, h = (int(stats[label, i]) for i in range(4))
        regions.append({"label": label, "peak": float(heat[component].max()), "mean": float(heat[component].mean()),
                        "area": area, "bbox_xyxy": [x, y, x + w, y + h]})
    regions.sort(key=lambda r: (r["peak"], r["area"]), reverse=True)
    kept = regions if max_regions <= 0 else regions[:max_regions]
    for rank, region in enumerate(kept, start=1):
        region["rank"] = rank
    mask = np.isin(labels, [r["label"] for r in kept]) if kept else np.zeros_like(valid, dtype=bool)
    return mask, {"rule": "hybrid_peak_fraction", "pixel_threshold_absolute": float(pixel_threshold),
                  "peak": peak, "peak_fraction": float(peak_fraction), "threshold_used": threshold,
                  "min_area": int(min_area), "regions_before": len(regions), "regions_kept": len(kept), "regions": kept}
