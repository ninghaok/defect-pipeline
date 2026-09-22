"""PBS experiment harness: immutable protocol producer/smoke, then independent full runs."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from detected_pipeline.config import load_project_config, roi_mask_for
from detected_pipeline.masks import external_gt, internal_mask
from detected_pipeline.roi import read_image
from detected_pipeline.util import atomic_write_json, sha256_file, utc_now
from prepare_simulation_streams import images, find_mask, link_or_copy
from export_simulation_table import export_category

CATEGORIES = ["qiumian_fupai", "qiumian_xiepai", "di_mian_detection", "wa_yuan_detection"]
WEIGHTS = ["dinov2_vitl14_pretrain.pth", "checkpoints_pro_angle.pth", "yolo26s-seg.pt"]


def command(*args, env=None):
    print("COMMAND", json.dumps([str(a) for a in args]), flush=True)
    subprocess.run([str(a) for a in args], cwd=PROJECT, env=env, check=True)


def verify_source():
    manifest = json.loads((PROJECT / "experiment_source.json").read_text())
    for item in manifest["files"]:
        if sha256_file(PROJECT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Source snapshot changed: {item['path']}")
    return manifest


def prepare(base):
    model_root = base / "models"; model_root.mkdir()
    with zipfile.ZipFile("/gdata1/ninghao/pipeline_models.zip") as archive:
        for name in WEIGHTS:
            with archive.open("models/" + name) as source, (model_root / name).open("xb") as target:
                shutil.copyfileobj(source, target)
            (PROJECT / "models" / name).symlink_to(model_root / name)
    # Ultralytics' AMP self-check may use the detection checkpoint; keep it local.
    detection = Path("/gdata1/huangjd/code/detected/yolo26n.pt")
    if detection.is_file():
        (PROJECT / "yolo26n.pt").symlink_to(detection)
    atomic_write_json(base / "weights.json", {name: sha256_file(model_root / name) for name in WEIGHTS})
    protocol = base / "protocol"
    command(sys.executable, PROJECT / "cli/prepare_simulation_streams.py", "--scenario", "all_data_lifecycle",
            "--output-root", protocol, "--dataset-root", "/gdata1/ninghao/dataset_523", "--reserve", "qiumian_xiepai=32,100,100", "--stream-mode", "stratified")
    manifest = json.loads((protocol / "stream_manifest.json").read_text())
    audit = []
    for category in manifest["classes"]:
        roles = {}
        for role, paths in [("reference", category["initial_reference_ok"]), ("bank", category["initial_bank_ok"]),
                            ("calibration", category["initial_calibration_ok"]),
                            ("stream", [str(p) for label in ("OK", "NG") for p in images(Path(category["stream_dir"]) / label)]),
                            ("test", [r["image"] for r in category["fixed_test"]])]:
            for path in paths:
                digest = sha256_file(Path(path))
                if digest in roles:
                    raise RuntimeError(f"Duplicate across protocol roles: {path}: {roles[digest]} / {role}")
                roles[digest] = role
        masks = []
        for image in images(Path(category["stream_dir"]) / "NG"):
            masks.append((image, find_mask(image.parent.parent / "mask", image)))
        masks.extend((Path(row["image"]), Path(row["mask"]) if row["mask"] else None) for row in category["fixed_test"] if row["label"] == "NG")
        mask_records = []
        for image, mask in masks:
            if mask is None: raise RuntimeError(f"Missing NG mask: {image}")
            gt = external_gt(mask)
            if gt.shape != read_image(image).shape[:2] or not gt.any(): raise RuntimeError(f"Invalid NG mask: {mask}")
            mask_records.append({"image": str(image), "mask": str(mask), "sha256": sha256_file(mask)})
        expected = category["ok"] + category["selected_ng"]
        audit.append({"category": category["category"], "expected_stream": expected,
                      "stream_ok": category["ok"], "stream_ng": category["selected_ng"],
                      "expected_batches": (expected + (99 if category["category"] == "qiumian_xiepai" else 199)) // (100 if category["category"] == "qiumian_xiepai" else 200),
                      "fixed_test_ok": sum(r["label"] == "OK" for r in category["fixed_test"]),
                      "fixed_test_ng": sum(r["label"] == "NG" for r in category["fixed_test"]),
                      "content_roles": roles, "ng_masks": mask_records})
    atomic_write_json(base / "protocol_audit.json", {"status": "ok", "manifest_sha256": sha256_file(protocol / "stream_manifest.json"), "categories": audit})
    return manifest


def smoke_view(base, manifest):
    category = next(r for r in manifest["classes"] if r["category"] == "wa_yuan_detection")
    smoke_root = base / "smoke_protocol"
    source = Path(category["stream_dir"])
    for label in ("OK", "NG"):
        available = images(source / label)
        if label == "NG":
            roi_path = roi_mask_for(load_project_config(PROJECT), category["category"])
            valid = []
            for path in available:
                gt = external_gt(find_mask(source / "mask", path))
                if roi_path: gt &= internal_mask(roi_path, gt.shape)
                if gt.any(): valid.append(path)
            available = valid
        selected = available[:50]
        if len(selected) < 50: raise RuntimeError("Smoke protocol requires 50 OK and 50 NG")
        for path in selected:
            link_or_copy(path, smoke_root / "streams" / category["category"] / label / path.name)
            if label == "NG":
                mask = find_mask(source / "mask", path)
                link_or_copy(mask, smoke_root / "streams" / category["category"] / "mask" / mask.name)
    row = dict(category, ok=50, selected_ng=50, stream_dir=str(smoke_root / "streams" / category["category"]),
               fixed_test=[r for label in ("OK", "NG") for r in [x for x in category["fixed_test"] if x["label"] == label][:4]])
    output = smoke_root / "stream_manifest.json"
    atomic_write_json(output, {**manifest, "classes": [row]})
    return output, row


def run_category(base, run_root, category, manifest_path, row, smoke):
    env = {**os.environ, "PIPELINE_RESULTS_ROOT": str(run_root),
           "PIPELINE_PRETRAINED_CACHE_ROOT": str(run_root / "pretrained_cache"),
           "PIPELINE_PRETRAINED_RESULTS_ROOT": str(run_root / "pretrained_artifacts")}
    os.environ.update(env)
    config = load_project_config(PROJECT)
    if smoke:
        config["training"]["epochs"] = 1
        config["review_sampling"]["enabled"] = False
        config["review_batch_size"] = 80
    atomic_write_json(run_root / "resolved_config.json", config)
    # Same CLI and control flow; smoke changes only the explicitly saved experimental settings.
    import run_lifecycle
    run_lifecycle.load_project_config = lambda _: config
    sys.argv = ["run_lifecycle.py", "--category", category, "--inbox", row["stream_dir"],
                "--initialization-manifest", str(manifest_path), "--batch-id", "smoke" if smoke else "table42"]
    run_lifecycle.main()
    workspace = run_root / "workspace"
    result = export_category(workspace, category, config, run_root / "summary")
    with sqlite3.connect(workspace / "state" / "pipeline.sqlite3") as db:
        processed = db.execute("SELECT COUNT(*) FROM samples WHERE category=? AND camera_id!='initialization'", (category,)).fetchone()[0]
        pending = db.execute("SELECT COUNT(*) FROM samples WHERE category=? AND reviewed_decision IS NULL", (category,)).fetchone()[0]
    expected = row["ok"] + row["selected_ng"]
    batch_size = 80 if smoke else 100 if category == "qiumian_xiepai" else 200
    if processed != expected or pending or result["pending_batches"] or result["completed_rows"] != (expected + batch_size - 1) // batch_size:
        raise RuntimeError(f"Incomplete lifecycle: expected={expected}, processed={processed}, pending={pending}, table={result}")
    summaries = list((workspace / "model_registry" / category / "milestones").glob("*/summary.json"))
    if smoke and not summaries: raise RuntimeError("Smoke did not train any YOLO candidate")
    for path in summaries:
        summary = json.loads(path.read_text())
        metadata = json.loads((workspace / "model_registry" / category / "versions" / summary["model_version"] / "model.json").read_text())
        if not metadata.get("fixed_test") or sha256_file(Path(metadata["checkpoint"])) != metadata["sha256"]:
            raise RuntimeError(f"Incomplete model artifact: {summary['model_version']}")
    return {**result, "expected_stream": expected, "processed": processed, "yolo_versions": len(summaries)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", *CATEGORIES], required=True)
    args = parser.parse_args(); base = args.base.resolve()
    run_root = base / "runs" / args.stage
    run_root.mkdir(parents=True, exist_ok=False)
    status = {"stage": args.stage, "started_at": utc_now(), "status": "running", "pbs_job_id": os.environ.get("PBS_JOBID")}
    atomic_write_json(run_root / "status.json", status)
    try:
        provenance = verify_source()
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("PBS container has no CUDA device")
        versions = {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "timm", "ultralytics", "scipy"]}
        atomic_write_json(run_root / "environment.json", {"python": sys.executable, "versions": versions,
                          "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "source": provenance})
        if args.stage == "smoke":
            manifest = prepare(base)
            manifest_path, row = smoke_view(base, manifest)
            result = run_category(base, run_root, row["category"], manifest_path, row, True)
            atomic_write_json(base / "smoke_ready.json", {"status": "ok", "protocol_sha256": sha256_file(base / "protocol/stream_manifest.json"),
                                                        "source_identity": provenance["identity"], "result": result})
        else:
            ready = json.loads((base / "smoke_ready.json").read_text())
            manifest_path = base / "protocol/stream_manifest.json"
            if ready["status"] != "ok" or ready["source_identity"] != provenance["identity"] or ready["protocol_sha256"] != sha256_file(manifest_path):
                raise RuntimeError("Producer smoke/protocol validation failed")
            for name, digest in json.loads((base / "weights.json").read_text()).items():
                if sha256_file(PROJECT / "models" / name) != digest: raise RuntimeError(f"Weight changed: {name}")
            manifest = json.loads(manifest_path.read_text())
            row = next(r for r in manifest["classes"] if r["category"] == args.stage)
            result = run_category(base, run_root, args.stage, manifest_path, row, False)
        status.update(status="completed", result=result, finished_at=utc_now())
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=utc_now())
        traceback.print_exc()
        raise
    finally:
        atomic_write_json(run_root / "status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__": main()
