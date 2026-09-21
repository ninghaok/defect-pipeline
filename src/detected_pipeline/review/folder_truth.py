from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from detected_pipeline.contracts import Decision, ReviewRecord
from detected_pipeline.util import atomic_write_json


class FolderGroundTruthReviewProvider:
    """Simulation-only reviewer that reads hidden truth from OK/NG path components."""

    review_all = True

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.ok_names = {str(x).casefold() for x in config.get("ok_directory_names", ["OK"])}
        self.ng_names = {str(x).casefold() for x in config.get("ng_directory_names", ["NG"])}
        self.mask_dir_names = tuple(str(x) for x in config.get("mask_directory_names", ["mask", "masks"]))
        self.mask_stem_suffixes = tuple(str(x) for x in config.get("mask_stem_suffixes", ["", "_t", "_mask"]))
        self.mask_extensions = tuple(str(x).lower() for x in config.get(
            "mask_extensions", [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]
        ))
        self.category_directory_names = {
            str(category): {str(name).casefold() for name in names}
            for category, names in config.get("category_directory_names", {}).items()
        }

    def is_ground_truth_artifact(self, path: Path) -> bool:
        mask_names = {name.casefold() for name in self.mask_dir_names}
        return any(part.casefold() in mask_names for part in path.parts)

    def _truth(self, image_path: Path) -> Decision:
        parts = {part.casefold() for part in image_path.parts}
        is_ok = bool(parts & self.ok_names)
        is_ng = bool(parts & self.ng_names)
        if is_ok == is_ng:
            raise ValueError(
                f"source path must contain exactly one OK/NG directory component: {image_path}"
            )
        return Decision.NG if is_ng else Decision.OK

    def _validate_category(self, category: str, image_path: Path) -> None:
        expected = self.category_directory_names.get(category, {category.casefold()})
        if not ({part.casefold() for part in image_path.parts} & expected):
            raise ValueError(
                f"source path does not match category {category}; expected one of {sorted(expected)}: {image_path}"
            )

    def _find_mask(self, image_path: Path) -> Path | None:
        matches: list[Path] = []
        for ancestor in image_path.parents:
            for directory_name in self.mask_dir_names:
                mask_root = ancestor / directory_name
                if not mask_root.is_dir():
                    continue
                for stem_suffix in self.mask_stem_suffixes:
                    for extension in self.mask_extensions:
                        candidate = mask_root / f"{image_path.stem}{stem_suffix}{extension}"
                        if candidate.is_file():
                            matches.append(candidate.resolve())
            if matches:
                break
        unique = sorted(set(matches))
        if len(unique) > 1:
            raise ValueError(f"ambiguous ground-truth masks for {image_path}: {unique}")
        return unique[0] if unique else None

    def review_prediction(self, prediction, image_path: Path) -> ReviewRecord:
        self._validate_category(prediction.category, image_path)
        truth = self._truth(image_path)
        mask = self._find_mask(image_path) if truth == Decision.NG else None
        record = ReviewRecord(
            sample_id=prediction.sample_id,
            category=prediction.category,
            model_decision=prediction.final_decision,
            reviewed_decision=truth,
            label_source="folder_ground_truth",
            mask_path=str(mask) if mask else None,
            reviewer="automatic_folder_ground_truth",
            metadata={"source_path": str(image_path), "mask_found": mask is not None},
        )
        self.validate_review_record(record)
        return record

    def export_review_batch(self, records: Iterable[dict[str, Any]], output: Path) -> Path:
        atomic_write_json(output, {"review_provider": "folder_ground_truth", "records": list(records)})
        return output

    def import_review_result(self, path: Path) -> list[ReviewRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: list[ReviewRecord] = []
        for row in payload["records"]:
            image_path = Path(row["source_path"])
            self._validate_category(row["category"], image_path)
            truth = self._truth(image_path)
            mask = self._find_mask(image_path) if truth == Decision.NG else None
            record = ReviewRecord(
                sample_id=row["sample_id"], category=row["category"],
                model_decision=Decision(row["final_decision"]), reviewed_decision=truth,
                label_source="folder_ground_truth", mask_path=str(mask) if mask else None,
                reviewer="automatic_folder_ground_truth",
                metadata={"source_path": str(image_path), "mask_found": mask is not None},
            )
            self.validate_review_record(record)
            result.append(record)
        return result

    def validate_review_record(self, record: ReviewRecord) -> None:
        if record.label_source != "folder_ground_truth":
            raise ValueError("folder reviewer must use label_source=folder_ground_truth")
        if record.reviewer != "automatic_folder_ground_truth":
            raise ValueError("invalid folder ground-truth reviewer")
