"""Bounded, restart-safe training-only augmentation using the SeaS JSON service.

One lifecycle process owns a category workspace. A lock rejects concurrent writers.
Failed/interrupted generation is never retried automatically or silently ignored.
"""
from __future__ import annotations

import json
import math
import subprocess
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from detected_pipeline.masks import internal_mask
from detected_pipeline.roi import read_image
from detected_pipeline.util import atomic_write_json, sha256_file
from .seas_contract import (TRUSTED, feedback_identity, identity, observation_index,
                            support_record, validate_generated)


def validate_policy(policy):
    integers = ("seed", "min_real_train_ng", "min_true_ok", "max_support_ng", "max_support_ok",
                "refresh_new_real_ng", "max_events", "max_train_synthetic")
    for key in integers:
        if type(policy.get(key)) is not int or policy[key] < (0 if key == "seed" else 1):
            raise ValueError(f"synthetic.{key}: expected a positive integer (seed may be zero)")
    if not (20 <= policy["min_true_ok"] <= policy["max_support_ok"] <= 60) or policy["max_support_ok"] % 2:
        raise ValueError("SeaS needs 20..60 true OK with an even maximum")
    if policy["min_real_train_ng"] < 5 or policy["max_support_ng"] < policy["min_real_train_ng"]:
        raise ValueError("SeaS needs >=5 real train NG and a sufficient support cap")
    if policy["seed"] >= 2**32:
        raise ValueError("synthetic.seed must be below 2**32")
    for key in ("max_synthetic_per_real_ng", "max_synthetic_fraction", "max_positive_fraction_increase"):
        value = policy.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value < 1:
            raise ValueError(f"synthetic.{key} must be within (0,1)")
    if not isinstance(policy.get("area_match_factor"), (int, float)) or not 1 <= policy["area_match_factor"] <= 10:
        raise ValueError("area_match_factor must be in [1,10]")
    command = policy.get("command")
    if not isinstance(command, list) or len(command) < 2 or not all(isinstance(x, str) for x in command):
        raise ValueError("Synthetic augmentation is enabled: configure the backend with cli/configure_synthetic.py and PIPELINE_SYNTHETIC_CONFIG, or explicitly select configs/synthetic_baseline.yaml. synthetic.command must be [absolute Python, absolute service.py]")
    if len(command) != 2 or not all(Path(x).is_absolute() and Path(x).is_file() for x in command):
        raise ValueError("synthetic.command requires exactly two existing absolute files")
    if not policy.get("reuse_root") or not Path(policy["reuse_root"]).is_absolute():
        raise ValueError("synthetic.reuse_root must be absolute")
    pinned = policy.get("backend_sha256", {})
    required = {str(p.resolve()) for p in Path(command[1]).parent.glob("*.py")}
    if not required or required != set(pinned):
        raise ValueError("Pin every backend sibling .py file, and only those files")
    if any(sha256_file(Path(p)) != digest for p, digest in pinned.items()):
        raise ValueError("Synthetic backend source changed since policy was frozen")
    # Retain the audited service's fixed, bounded formal generation protocol.
    allowed = {"threshold", "inference_steps", "guidance_scale"}
    if set(policy.get("generation_config", {})) - allowed:
        raise ValueError("Only threshold/inference_steps/guidance_scale may override formal SeaS settings")


def quota(real_ng, train_ok, policy):
    """R=ROI-effective real NG, O=all YOLO training OK; every bound must hold."""
    total = real_ng + train_ok
    if not real_ng or not total:
        return 0
    p = policy["max_synthetic_fraction"]
    delta = policy["max_positive_fraction_increase"]
    bounds = [policy["max_train_synthetic"], policy["max_synthetic_per_real_ng"] * real_ng,
              p * total / (1 - p)]
    denominator = 1 - real_ng / total - delta
    if denominator > 0:
        bounds.append(delta * total / denominator)
    return max(0, math.floor(min(bounds) + 1e-10))


def coverage(image_path, mask_path, roi_mask):
    image = read_image(Path(image_path))
    mask = internal_mask(mask_path, image.shape[:2])
    roi = np.ones(mask.shape, dtype=bool)
    if roi_mask is not None:
        raw = read_image(Path(roi_mask), cv2.IMREAD_GRAYSCALE)
        roi = cv2.resize(raw, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST) >= 128
    return float(np.count_nonzero(mask & roi) / max(1, np.count_nonzero(roi)))


def select_by_area(items, real_areas, budget, factor, roi_mask):
    """Match real-area quantiles, never resize/alter labels to manufacture a match."""
    pool = [(coverage(r["image"], r["mask"], roi_mask), r) for r in items]
    pool = [(area, row) for area, row in pool if 0 < area < 1]
    chosen = []
    if budget <= 0:
        return chosen
    targets = np.quantile(real_areas, (np.arange(budget) + .5) / budget)
    for target in targets:
        eligible = [(abs(math.log(area / target)), row["image_sha256"], i)
                    for i, (area, row) in enumerate(pool) if target / factor <= area <= target * factor]
        if not eligible:
            continue
        _, _, index = min(eligible)
        area, row = pool.pop(index)
        chosen.append(dict(row, area_fraction=area, target_area_fraction=float(target)))
    return chosen


def pixel_digest(path):
    import hashlib
    image = read_image(Path(path))
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


@contextmanager
def category_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "writer.lock"
    # A leftover lock after SIGKILL requires explicit inspection, not concurrent retry.
    with path.open("x") as handle:
        handle.write("Single category writer; inspect running jobs before removing a stale lock.\n")
    try:
        yield
    finally:
        path.unlink()


def prepare(workspace, category, milestone, policy, train_ng, cal_ng, train_ok,
            calibration_ok, roi_mask, context):
    validate_policy(policy)
    if context is None:
        raise ValueError("Enabled augmentation requires lifecycle provenance and GPU-release context")
    root = Path(workspace) / "augmentation" / category
    with category_lock(root):
        return _prepare(root, workspace, category, milestone, policy, train_ng, cal_ng,
                        train_ok, calibration_ok, roi_mask, context)


def _prepare(root, workspace, category, milestone, policy, train_ng, cal_ng, train_ok,
             calibration_ok, roi_mask, context):
    policy_sha = identity(policy)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "policy_sha256": policy_sha, "events": [], "last_real_ng_sha256": []}
    if state["policy_sha256"] != policy_sha:
        raise ValueError("Policy changed inside a run; use a fresh result workspace")
    areas = {r["sha256"]: coverage(r["copy_path"], r["mask"], roi_mask) for r in train_ng}
    real = [r for r in train_ng if areas[r["sha256"]] > 0]
    hashes = {r["sha256"] for r in real}
    budget = quota(len(real), len(train_ok), policy)
    info = {"enabled": True, "policy_sha256": policy_sha, "real_train_ng_effective": len(real),
            "budget": budget, "selected": [], "selected_count": 0, "synthetic_fraction": 0.0,
            "event_count": len(state["events"]),
            "annotation_status": "synthetic_pseudo_mask", "production_approved": False}
    if state["events"] and state["events"][-1]["status"] != "completed":
        raise RuntimeError("Previous generation failed/interrupted; inspect event before an explicit new run")
    observations, batch, position = observation_index(workspace, category)
    true_ok = [r for r in train_ok if r.get("label_source") in TRUSTED]
    if (len(real) < policy["min_real_train_ng"] or len(true_ok) < policy["min_true_ok"] or budget == 0):
        return [], dict(info, reason="insufficient_real_support_or_quota")
    forbidden = set(context["forbidden_sha"]) | {r["sha256"] for r in cal_ng}
    forbidden |= {sha256_file(Path(p)) for p in calibration_ok}
    if forbidden & {r["sha256"] for r in real + train_ok}:
        raise ValueError("Training data overlap calibration/test/reference exclusion set")
    should_generate = not state["events"] or len(hashes - set(state["last_real_ng_sha256"])) >= policy["refresh_new_real_ng"]
    if should_generate and len(state["events"]) < policy["max_events"]:
        # Deterministic area-stratified support; caps do not preferentially retain huge defects.
        ordered = sorted(real, key=lambda r: (areas[r["sha256"]], r["sha256"]))
        indices = np.linspace(0, len(ordered) - 1, min(len(ordered), policy["max_support_ng"]), dtype=int)
        support_ng = [ordered[int(i)] for i in indices]
        support_ok = sorted(true_ok, key=lambda r: r["sha256"])[:policy["max_support_ok"]]
        support_ok = support_ok[:len(support_ok) // 2 * 2]
        event_id = f"event{len(state['events']) + 1:03d}_v{milestone}"
        request_path = root / "requests" / f"{event_id}.json"
        request = {"schema_version": 1, "mode": "formal", "category": category, "event_id": event_id,
                   "observed_batch": batch, "observed_stream_position": position, "train_version": f"v{milestone}",
                   "seed": policy["seed"], "generation_config": policy.get("generation_config", {}),
                   "forbidden_image_sha256": sorted(forbidden)}
        for key, rows, label in (("support_ng", support_ng, "NG"), ("support_ok", support_ok, "OK")):
            request[key] = [dict(support_record(r, label, observations, context["initial_bank_sha"]),
                                 category=category, coordinate_space="full_frame") for r in rows]
        if roi_mask is not None:
            request["roi"] = {"path": str(Path(roi_mask).resolve()), "sha256": sha256_file(roi_mask),
                              "convention": "white_is_selected", "fill_outside": 255, "coordinate_space": "full_frame"}
        atomic_write_json(request_path, request)
        event = {"event_id": event_id, "status": "running", "request": str(request_path),
                 "request_sha256": sha256_file(request_path), "output": str(root / "events" / event_id)}
        state["events"].append(event)
        # Persist attempt BEFORE starting; interruption cannot reset lifetime generation budget.
        atomic_write_json(state_path, state)
        before = feedback_identity(workspace)
        try:
            with context["release"]():
                with (root / f"{event_id}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(policy["command"] + ["--request", str(request_path), "--output", event["output"],
                                   "--reuse-root", policy["reuse_root"]], stdout=log, stderr=subprocess.STDOUT, check=True)
            if feedback_identity(workspace) != before:
                raise RuntimeError("Generator modified feedback or split tables")
            manifest_path = Path(event["output"]) / "manifest.json"
            result = json.loads(manifest_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed" or result.get("request_sha256") != event["request_sha256"]:
                raise ValueError("Backend did not complete the requested event")
            if result.get("candidate_count", 0) > 120 or len(result.get("items", [])) > 40:
                raise ValueError("Backend exceeded audited per-event generation cap")
            event.update(status="completed", manifest_sha256=sha256_file(manifest_path),
                         candidate_count=result.get("candidate_count"), output_count=len(result.get("items", [])))
            state["last_real_ng_sha256"] = sorted(hashes)
            atomic_write_json(state_path, state)
        except BaseException as exc:
            event.update(status="failed", error=str(exc))
            atomic_write_json(state_path, state)
            raise
    event = state["events"][-1]
    request_path = Path(event["request"])
    manifest_path = Path(event["output"]) / "manifest.json"
    if sha256_file(request_path) != event["request_sha256"] or sha256_file(manifest_path) != event["manifest_sha256"]:
        raise ValueError("Cached generation provenance changed")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    available = {r["sha256"] for r in real + train_ok}
    if not {r["image_sha256"] for r in request["support_ng"] + request["support_ok"]} <= available:
        return [], dict(info, reason="cached_support_no_longer_in_current_training_set")
    for row in request["support_ng"] + request["support_ok"]:
        for key in (("image", "mask") if row["label"] == "NG" else ("image",)):
            if sha256_file(Path(row[key])) != row[key + "_sha256"]:
                raise ValueError("Cached generation support changed")
    _, items = validate_generated(manifest_path, request_path, forbidden | available,
                                  policy["allow_structurally_valid_unreviewed"])
    # Encoded-file hashes alone miss identical pixels saved under another encoding.
    seen_pixels = {pixel_digest(r["copy_path"]) for r in real + train_ok}
    seen_pixels |= set(context.get("forbidden_pixel_sha", []))
    seen_pixels |= {pixel_digest(r["copy_path"]) for r in cal_ng}
    deduped = []
    for row in items:
        key = pixel_digest(row["image"])
        if key not in seen_pixels:
            deduped.append(row)
            seen_pixels.add(key)
    selected = select_by_area(deduped, list(areas[r["sha256"]] for r in real), budget,
                              policy["area_match_factor"], roi_mask)
    info.update(selected=selected, selected_count=len(selected), event=event,
                event_count=len(state["events"]), synthetic_fraction=len(selected) / (len(real) + len(train_ok) + len(selected)),
                reason="selected" if selected else "quality_or_area_gate_selected_zero")
    return [(Path(r["image"]), Path(r["mask"])) for r in selected], info
