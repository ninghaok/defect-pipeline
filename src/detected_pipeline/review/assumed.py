from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from detected_pipeline.contracts import Decision, ReviewRecord
from detected_pipeline.util import atomic_write_json


class ModelAssumedReviewProvider:
    """仿真审核器：明确把模型结论复制为审核结论，不冒充人工真值。"""

    def export_review_batch(self, records: Iterable[dict[str, Any]], output: Path) -> Path:
        atomic_write_json(output, {"review_provider": "model_assumed_review", "records": list(records)})
        return output

    def import_review_result(self, path: Path) -> list[ReviewRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = []
        for row in payload["records"]:
            decision = Decision(row["final_decision"])
            result.append(ReviewRecord(
                sample_id=row["sample_id"], category=row["category"],
                model_decision=decision, reviewed_decision=decision,
                label_source="model_assumed_review",
                mask_path=row.get("binary_mask_path") if decision == Decision.NG else None,
            ))
        return result

    def validate_review_record(self, record: ReviewRecord) -> None:
        if record.label_source != "model_assumed_review":
            raise ValueError("assumed reviewer must use label_source=model_assumed_review")
        if record.model_decision != record.reviewed_decision:
            raise ValueError("assumed reviewer cannot change the model decision")
        if record.reviewed_decision == Decision.NG and not record.mask_path:
            raise ValueError("NG review requires a mask")

    def review_prediction(self, prediction, image_path: Path | None = None) -> ReviewRecord:
        record = ReviewRecord(
            sample_id=prediction.sample_id, category=prediction.category,
            model_decision=prediction.final_decision,
            reviewed_decision=prediction.final_decision,
            label_source="model_assumed_review",
            mask_path=prediction.binary_mask_path if prediction.final_decision == Decision.NG else None,
        )
        self.validate_review_record(record)
        return record
