import numpy as np

from detected_pipeline.pretrained.segmentation import hybrid_regions


def _heat():
    heat = np.full((100, 100), 1.0, np.float32)
    heat[10:30, 10:30] = 5.0          # main defect
    heat[60:65, 60:65] = 4.5          # small secondary blob (25 px)
    heat[80:95, 5:20] = 3.0           # below 0.7 * peak
    return heat


def test_hybrid_rule_keeps_peak_region_only_when_relative_term_dominates():
    heat = _heat(); valid = np.ones_like(heat, bool)
    mask, stats = hybrid_regions(heat, valid, pixel_threshold=2.0, peak_fraction=0.7, min_area=50, max_regions=3)
    assert stats["threshold_used"] == 3.5 and stats["regions_kept"] == 1
    assert mask[15, 15] and not mask[85, 10] and not mask[62, 62]   # secondary blob dropped by min_area


def test_absolute_floor_prevents_flooding_flat_maps():
    heat = np.full((50, 50), 1.0, np.float32); heat[20:25, 20:25] = 1.2
    mask, stats = hybrid_regions(heat, np.ones_like(heat, bool), pixel_threshold=2.0, peak_fraction=0.5, min_area=1)
    assert not mask.any() and stats["regions_kept"] == 0 and stats["threshold_used"] == 2.0


def test_roi_limits_regions_and_ranks_by_peak():
    heat = _heat(); valid = np.ones_like(heat, bool); valid[:50] = False    # main defect outside the ROI
    mask, stats = hybrid_regions(heat, valid, pixel_threshold=2.0, peak_fraction=0.5, min_area=1, max_regions=3)
    assert stats["peak"] == 4.5 and stats["regions"][0]["peak"] == 4.5 and stats["regions"][1]["peak"] == 3.0
    assert not mask[15, 15] and mask[62, 62]
