from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Decision(str, Enum):
    OK = "OK"
    NG = "NG"


@dataclass(frozen=True)
class InferenceContext:
    sample_id: str
    category: str
    batch_id: str = "default"
    camera_id: str = "unknown"
    timestamp: str | None = None


@dataclass(frozen=True)
class PretrainedPrediction:
    sample_id: str
    category: str
    feature_norm_score: float
    feature_norm_decision: Decision
    patchcore_score: float
    patchcore_decision: Decision
    final_decision: Decision
    is_boundary: bool
    is_conflict: bool
    pixel_score_map_path: str
    binary_mask_path: str
    thresholds_used: dict[str, float]
    latency_ms: float
    plugin_name: str
    plugin_version: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.sample_id or not self.category:
            raise ValueError("sample_id and category are required")
        for name in ("feature_norm_decision", "patchcore_decision", "final_decision"):
            if not isinstance(getattr(self, name), Decision):
                raise ValueError(f"{name} must be a Decision enum")
        for name in ("feature_norm_score", "patchcore_score", "latency_ms"):
            value = float(getattr(self, name))
            if value != value or (name == "latency_ms" and value < 0):
                raise ValueError(f"invalid {name}: {value}")
        if not isinstance(self.thresholds_used, dict) or not self.thresholds_used:
            raise ValueError("thresholds_used must be a non-empty mapping")
        for key, value in self.thresholds_used.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"threshold {key} is not numeric") from exc
            if numeric != numeric:
                raise ValueError(f"threshold {key} is NaN")
        for name in ("plugin_name", "plugin_version", "model_version"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        for name in ("pixel_score_map_path", "binary_mask_path"):
            path = Path(getattr(self, name))
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty prediction artifact: {path}")


@dataclass(frozen=True)
class SupervisedPrediction:
    sample_id: str
    category: str
    image_score: float
    image_decision: Decision
    pixel_probability_path: str
    binary_mask_path: str
    image_threshold: float
    pixel_threshold: float
    latency_ms: float
    checkpoint_sha256: str
    model_version: str


@dataclass(frozen=True)
class ReviewRecord:
    sample_id: str
    category: str
    model_decision: Decision
    reviewed_decision: Decision
    label_source: str
    mask_path: str | None = None
    reviewer: str = "model_assumed_review"
    metadata: dict[str, Any] = field(default_factory=dict)
