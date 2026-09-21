"""Sampled review of model-OK images.

Model-NG images are always reviewed.  Model-OK images are ranked by score: the top ``high_fraction`` are
all reviewed; the middle ``middle_fraction`` and the lowest remainder are spot-checked at ``middle_rate`` /
``low_rate`` (with minimum counts).  A segment whose spot check finds an NG is escalated to full review.
Unreviewed images of a passed segment become pseudo-OK (training OK only, never calibration).
"""
from __future__ import annotations

import random
from typing import Any, Callable

SEGMENTS = ("high", "middle", "low")


def plan_review(scores: list[float], config: dict[str, Any], seed: str) -> dict[str, Any]:
    """Split model-OK images (by index) into segments and pick the spot-check sample of each segment."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n = len(order); high_n = round(n * float(config.get("high_fraction", 0.2)))
    middle_n = round(n * float(config.get("middle_fraction", 0.4)))
    segments = {"high": order[:high_n], "middle": order[high_n:high_n + middle_n], "low": order[high_n + middle_n:]}
    rng = random.Random(seed); sampled = {"high": list(segments["high"])}
    for name, rate_key, min_key in (("middle", "middle_rate", "middle_min"), ("low", "low_rate", "low_min")):
        members = segments[name]
        count = min(len(members), max(int(config.get(min_key, 0)), round(len(members) * float(config.get(rate_key, 0.0)))))
        sampled[name] = sorted(rng.sample(members, count)) if count else []
    return {"segments": segments, "sampled": sampled}


def run_sampled_review(scores: list[float], config: dict[str, Any], seed: str, review: Callable[[int], bool]) -> dict[str, Any]:
    """Review the planned samples; escalate a segment to full review when a spot check finds an NG.

    ``review(index)`` performs the (simulated) manual review and returns True when the image is NG.
    Returns reviewed indices, pseudo-OK indices and per-segment statistics.
    """
    plan = plan_review(scores, config, seed)
    reviewed: dict[int, bool] = {}; stats = {}
    for name in SEGMENTS:
        members = plan["segments"][name]; sample = plan["sampled"][name]
        found_ng = False
        for index in sample:
            reviewed[index] = review(index); found_ng |= reviewed[index]
        escalated = found_ng and len(sample) < len(members)
        if escalated:
            for index in members:
                if index not in reviewed:
                    reviewed[index] = review(index)
        stats[name] = {"members": len(members), "sampled": len(sample), "ng_in_sample": int(sum(reviewed[i] for i in sample)),
                       "escalated": escalated, "reviewed": int(sum(i in reviewed for i in members)),
                       "ng_found": int(sum(reviewed.get(i, False) for i in members))}
    pseudo_ok = [i for i in range(len(scores)) if i not in reviewed]
    return {"reviewed": reviewed, "pseudo_ok": pseudo_ok, "stats": stats}
