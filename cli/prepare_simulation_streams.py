from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detected_pipeline.config import load_yaml
from detected_pipeline.util import atomic_write_json


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS)


_DIGESTS: dict[str, str] = {}


def digest_of(path: Path) -> str:
    import hashlib
    key = str(path)
    if key not in _DIGESTS:
        _DIGESTS[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _DIGESTS[key]


def unique_images(paths: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    seen = set(); result = []
    for split, path in paths:
        digest = digest_of(path)
        if digest not in seen:
            seen.add(digest); result.append((split, path))
    return result


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def find_mask(mask_root: Path, image: Path) -> Path | None:
    for suffix in ("_t", "_mask", ""):
        for extension in sorted(EXTENSIONS):
            candidate = mask_root / f"{image.stem}{suffix}{extension}"
            if candidate.is_file():
                return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic realistic or accelerated stream views")
    parser.add_argument("--scenario", choices=("simplified_lifecycle", "all_data_lifecycle"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--reference-ok", type=int, default=32)
    parser.add_argument("--calibration-ok", type=int, default=200)
    parser.add_argument("--bank-ok", type=int, default=200, help="OK reserved for the pretrained memory bank (also YOLO training OK); shrinks per category")
    parser.add_argument("--min-stream-ok", type=int, default=100)
    parser.add_argument("--reserve", action="append", default=[], metavar="CATEGORY=REF,CAL,BANK",
                        help="per-category override of reference/calibration/bank OK counts, e.g. qiumian_xiepai=32,100,100")
    parser.add_argument("--stream-mode", choices=("random", "stratified"), default="stratified",
                        help="random: one global shuffle at the target ratio; stratified: every batch has the same NG rate")
    parser.add_argument("--print-full-manifest", action="store_true",
                        help="Print every selected path; by default only a compact summary is printed.")
    parser.add_argument(
        "--ng-ratio", action="append", default=[], metavar="CATEGORY=RATIO",
        help="NG fraction of the stream for a category, e.g. qiumian_fupai=0.10, or 'all' (every available NG; default). May be repeated.",
    )
    args = parser.parse_args()
    config = load_yaml(PROJECT / "configs/simulation.yaml")
    dataset = args.dataset_root or Path(config["dataset_root"])
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    seed = int(config.get("seed", 42))
    ratios = {}   # default: every available NG; simulation.yaml realistic_ng_ratios are reference values only
    for item in args.ng_ratio:
        if "=" not in item:
            parser.error(f"invalid --ng-ratio {item!r}; expected CATEGORY=RATIO")
        category, raw = item.split("=", 1)
        if category not in config["category_directories"]:
            parser.error(f"unknown category in --ng-ratio: {category}")
        if raw.strip().lower() in {"all", "-1"}:
            ratios[category] = "all"; continue
        try:
            ratio = float(raw)
        except ValueError:
            parser.error(f"invalid NG ratio for {category}: {raw}")
        if not 0.0 <= ratio < 1.0:
            parser.error(f"NG ratio must be in [0,1) or 'all': {category}={ratio}")
        ratios[category] = ratio
    reserves = {}
    for item in args.reserve:
        if "=" not in item or item.split("=", 1)[1].count(",") != 2:
            parser.error(f"invalid --reserve {item!r}; expected CATEGORY=REF,CAL,BANK")
        category, raw = item.split("=", 1)
        if category not in config["category_directories"]:
            parser.error(f"unknown category in --reserve: {category}")
        reserves[category] = tuple(int(v) for v in raw.split(","))
    scenario_root = args.output_root.resolve()
    stream_root = scenario_root / "streams"
    summaries = []

    for category, source_name in config["category_directories"].items():
        source = dataset / source_name
        if args.scenario in {"simplified_lifecycle", "all_data_lifecycle"}:
            # the dataset's test split is the FIXED TEST SET: never streamed, trained, calibrated on or used to select anything
            test_ok = unique_images([("test", p) for p in images(source / "test" / "OK")])
            test_ng = unique_images([("test", p) for p in images(source / "test" / "NG")])
            test_digests = {digest_of(p) for _, p in test_ok + test_ng}
            ok_tagged = [(s_, p) for s_, p in unique_images([(split, p) for split in ("train", "val") for p in images(source / split / "OK")]) if digest_of(p) not in test_digests]
            ng_tagged = [(s_, p) for s_, p in unique_images([(split, p) for split in ("train", "val") for p in images(source / split / "NG")]) if digest_of(p) not in test_digests]
            # memory-bank OK shrink per category so that at least --min-stream-ok unique OK remain for the stream
            n_ref, n_cal, n_bank = reserves.get(category, (args.reference_ok, args.calibration_ok, args.bank_ok))
            bank_count = min(n_bank, max(0, len(ok_tagged) - n_ref - n_cal - args.min_stream_ok))
            reserve_total = n_ref + n_cal + bank_count
            if len(ok_tagged) < reserve_total + args.min_stream_ok:
                raise ValueError(f"{category}: need {reserve_total + args.min_stream_ok} unique OK, found {len(ok_tagged)}")
            rng = random.Random(f"{seed}:{category}:initialization")
            shuffled_ok = list(ok_tagged); rng.shuffle(shuffled_ok)
            reference = sorted(shuffled_ok[:n_ref], key=lambda x: str(x[1]))
            calibration = sorted(shuffled_ok[n_ref:n_ref + n_cal], key=lambda x: str(x[1]))
            bank = sorted(shuffled_ok[n_ref + n_cal:reserve_total], key=lambda x: str(x[1]))
            reserved = {str(p.resolve()) for _, p in reference + calibration + bank}
            ok_tagged = [(s, p) for s, p in ok_tagged if str(p.resolve()) not in reserved]
            ok = [p for _, p in ok_tagged]; ng = [p for _, p in ng_tagged]
        else:
            reference = calibration = bank = []; test_ok = test_ng = []
            ok_tagged = [("test", p) for p in images(source / "test/OK")]
            ng_tagged = [("test", p) for p in images(source / "test/NG")]
            ok = [p for _, p in ok_tagged]; ng = [p for _, p in ng_tagged]
        target = ratios.get(category, "all")
        if args.scenario == "simplified_lifecycle" and target != "all":
            desired_ng = min(round(len(ok) * target / (1.0 - target)), len(ng))
            rng = random.Random(f"{seed}:{category}:realistic_ng")
            selected_ng = sorted(rng.sample(ng, desired_ng))
        else:
            target = len(ng) / (len(ok) + len(ng)) if ok or ng else 0.0
            selected_ng = ng

        category_root = stream_root / category
        methods = {"hardlink": 0, "copy": 0, "existing": 0}
        split_by_path = {str(p): split for split, p in ok_tagged + ng_tagged}
        for label, selected in (("OK", ok), ("NG", selected_ng)):
            for source_image in selected:
                prefix = split_by_path.get(str(source_image), "source")
                name = f"{prefix}__{source_image.name}" if args.scenario in {"simplified_lifecycle", "all_data_lifecycle"} else source_image.name
                method = link_or_copy(source_image, category_root / label / name)
                methods[method] += 1
        found_masks = 0
        for source_image in selected_ng:
            mask = find_mask(source / "mask", source_image)
            if mask:
                prefix = split_by_path.get(str(source_image), "source")
                name = f"{prefix}__{mask.name}" if args.scenario in {"simplified_lifecycle", "all_data_lifecycle"} else mask.name
                method = link_or_copy(mask, category_root / "mask" / name)
                methods[method] += 1
                found_masks += 1
        actual = len(selected_ng) / (len(ok) + len(selected_ng)) if ok or selected_ng else 0.0
        summaries.append({
            "category": category, "source_category": source_name, "ok": len(ok),
            "available_ng": len(ng), "selected_ng": len(selected_ng),
            "target_ng_ratio": target, "actual_ng_ratio": actual,
            "masks_found": found_masks, "stream_dir": str(category_root),
            "initial_reference_ok": [str(p) for _, p in reference],
            "initial_calibration_ok": [str(p) for _, p in calibration],
            "initial_bank_ok": [str(p) for _, p in bank],
            "fixed_test": [{"image": str(p), "label": "OK", "mask": None} for _, p in test_ok]
                        + [{"image": str(p), "label": "NG", "mask": (str(find_mask(source / "mask", p)) if find_mask(source / "mask", p) else None)} for _, p in test_ng],
        })

    manifest = {"scenario": args.scenario, "seed": seed, "dataset_root": str(dataset), "stream_mode": args.stream_mode,
                "requested_ng_ratios": ratios,
                "classes": summaries}
    manifest_path = scenario_root / "stream_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError(f"existing stream manifest differs; use a new output directory: {manifest_path}")
    else:
        atomic_write_json(manifest_path, manifest)
    if args.print_full_manifest:
        output = manifest
    else:
        output = {
            "scenario": manifest["scenario"], "seed": seed,
            "dataset_root": str(dataset), "manifest_path": str(manifest_path),
            "classes": [{
                "category": row["category"], "ok": row["ok"],
                "selected_ng": row["selected_ng"], "available_ng": row["available_ng"],
                "target_ng_ratio": row["target_ng_ratio"], "actual_ng_ratio": row["actual_ng_ratio"],
                "masks_found": row["masks_found"],
                "reference_ok": len(row["initial_reference_ok"]),
                "calibration_ok": len(row["initial_calibration_ok"]),
                "bank_ok": len(row["initial_bank_ok"]),
                "fixed_test_ok": sum(t["label"] == "OK" for t in row["fixed_test"]),
                "fixed_test_ng": sum(t["label"] == "NG" for t in row["fixed_test"]),
            } for row in summaries],
        }
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
