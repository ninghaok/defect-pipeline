from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from detected_pipeline.contracts import Decision, InferenceContext, PretrainedPrediction, ReviewRecord
from detected_pipeline.util import append_jsonl, atomic_copy, atomic_write_json, sha256_file, utc_now

POOLS = (
    "raw_pool", "ok_pending_pool", "alarm_candidate_pool", "boundary_hard_pool",
    "model_conflict_pool", "confirmed_ok_pool", "confirmed_ng_pool",
    "historical_false_positive_pool", "historical_false_negative_pool",
    "labeled_pool", "annotation_qc_failed_pool",
    "sampling_pseudo_ok_pool",
)


class _ClosingConnection(sqlite3.Connection):
    """Commit/rollback like sqlite3.Connection, then release the file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class FeedbackStore:
    def __init__(self, workspace: Path, categories: list[str]):
        self.workspace = workspace
        self.categories = categories
        self.db_path = workspace / "state" / "pipeline.sqlite3"
        self.events_path = workspace / "state" / "events.jsonl"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        for category in categories:
            for pool in POOLS:
                (workspace / "data" / category / pool).mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            self._migrate_global_sha_unique(db)
            db.executescript("""
                CREATE TABLE IF NOT EXISTS samples(
                    sample_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, category TEXT NOT NULL,
                    source_path TEXT NOT NULL, batch_id TEXT NOT NULL, camera_id TEXT NOT NULL,
                    model_decision TEXT, reviewed_decision TEXT, label_source TEXT,
                    mask_path TEXT, created_at TEXT NOT NULL, UNIQUE(category,sha256)
                );
                CREATE TABLE IF NOT EXISTS pool_membership(
                    sample_id TEXT NOT NULL, pool TEXT NOT NULL, copy_path TEXT NOT NULL,
                    created_at TEXT NOT NULL, PRIMARY KEY(sample_id,pool),
                    FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
                );
                CREATE TABLE IF NOT EXISTS category_state(
                    category TEXT PRIMARY KEY, last_trained_ng INTEGER NOT NULL DEFAULT 0,
                    active_dataset_version TEXT, active_model_version TEXT, pending_training_job TEXT
                );
                CREATE TABLE IF NOT EXISTS training_runs(
                    category TEXT NOT NULL, fingerprint TEXT NOT NULL, dataset_version TEXT NOT NULL,
                    status TEXT NOT NULL, job_id TEXT, model_version TEXT, created_at TEXT NOT NULL,
                    PRIMARY KEY(category,fingerprint)
                );
                CREATE TABLE IF NOT EXISTS dataset_split_assignment(
                    sample_id TEXT PRIMARY KEY, category TEXT NOT NULL,
                    decision TEXT NOT NULL, split TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
                );
            """)
            for category in self.categories:
                db.execute("INSERT OR IGNORE INTO category_state(category) VALUES (?)", (category,))

    @staticmethod
    def _migrate_global_sha_unique(db: sqlite3.Connection) -> None:
        row = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='samples'"
        ).fetchone()
        if not row or "sha256 TEXT UNIQUE" not in (row["sql"] or ""):
            return
        samples = [tuple(item) for item in db.execute("SELECT * FROM samples").fetchall()]
        memberships = [tuple(item) for item in db.execute("SELECT * FROM pool_membership").fetchall()]
        db.execute("PRAGMA foreign_keys=OFF")
        db.execute("DROP TABLE pool_membership")
        db.execute("DROP TABLE samples")
        db.executescript("""
            CREATE TABLE samples(
                sample_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, category TEXT NOT NULL,
                source_path TEXT NOT NULL, batch_id TEXT NOT NULL, camera_id TEXT NOT NULL,
                model_decision TEXT, reviewed_decision TEXT, label_source TEXT,
                mask_path TEXT, created_at TEXT NOT NULL, UNIQUE(category,sha256)
            );
            CREATE TABLE pool_membership(
                sample_id TEXT NOT NULL, pool TEXT NOT NULL, copy_path TEXT NOT NULL,
                created_at TEXT NOT NULL, PRIMARY KEY(sample_id,pool),
                FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
            );
        """)
        if samples:
            db.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)", samples)
        if memberships:
            db.executemany("INSERT INTO pool_membership VALUES (?,?,?,?)", memberships)
        db.execute("PRAGMA foreign_keys=ON")

    def _copy_to_pool(self, image: Path, sample_id: str, category: str, pool: str) -> Path:
        if pool not in POOLS:
            raise ValueError(f"unknown pool: {pool}")
        destination = self.workspace / "data" / category / pool / f"{sample_id}{image.suffix.lower()}"
        # Pool membership is logical metadata. Prefer a hard link so that the
        # sampling and oracle branches do not multiply the image disk usage.
        # Fall back to a real copy when source/destination are on different
        # volumes or the filesystem does not support hard links.
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(image, destination)
            except OSError:
                atomic_copy(image, destination)
        sidecar = destination.with_suffix(destination.suffix + ".json")
        atomic_write_json(sidecar, {"sample_id": sample_id, "category": category, "pool": pool})
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO pool_membership VALUES (?,?,?,?)",
                (sample_id, pool, str(destination), utc_now()),
            )
        return destination

    def ingest(self, image: Path, context: InferenceContext, prediction: PretrainedPrediction) -> bool:
        prediction.validate()
        digest = sha256_file(image)
        with self._connect() as db:
            existing = db.execute(
                "SELECT sample_id FROM samples WHERE category=? AND sha256=?", (context.category, digest)
            ).fetchone()
            if existing:
                return False
            db.execute(
                "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (context.sample_id, digest, context.category, str(image), context.batch_id,
                 context.camera_id, prediction.final_decision.value, None, None, None, utc_now()),
            )
        self._copy_to_pool(image, context.sample_id, context.category, "raw_pool")
        if prediction.final_decision == Decision.NG:
            self._copy_to_pool(image, context.sample_id, context.category, "alarm_candidate_pool")
        else:
            self._copy_to_pool(image, context.sample_id, context.category, "ok_pending_pool")
        if prediction.is_boundary:
            self._copy_to_pool(image, context.sample_id, context.category, "boundary_hard_pool")
        if prediction.is_conflict:
            self._copy_to_pool(image, context.sample_id, context.category, "model_conflict_pool")
        result_path = self.workspace / "inference_results" / context.category / f"{context.sample_id}.json"
        atomic_write_json(result_path, prediction.to_dict())
        append_jsonl(self.events_path, {"event": "sample_ingested", "at": utc_now(), **prediction.to_dict()})
        return True

    def contains(self, category: str, sha256: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM samples WHERE category=? AND sha256=?", (category, sha256)
            ).fetchone()
        return row is not None

    def seed_confirmed_ok_for_training(self, image: Path, category: str, sample_id: str) -> bool:
        """Add a trusted initialization OK and permanently lock it to YOLO train.

        These images may have calibrated the independent ADPretrain baseline,
        but are never online-stream, YOLO validation, threshold-calibration or
        fixed-test samples.
        """
        digest = sha256_file(image)
        now = utc_now()
        created = False
        effective_sample_id = sample_id
        with self._connect() as db:
            existing = db.execute(
                "SELECT sample_id,label_source FROM samples WHERE category=? AND sha256=?",
                (category, digest),
            ).fetchone()
            if existing:
                if existing["label_source"] != "initial_calibration_yolo_train_ok":
                    raise ValueError(
                        f"initial YOLO-train OK collides with a different role: {image} -> {dict(existing)}"
                    )
                db.execute(
                    "INSERT OR REPLACE INTO dataset_split_assignment VALUES (?,?,?,?,?)",
                    (existing["sample_id"], category, "OK", "train_locked", now),
                )
                effective_sample_id = existing["sample_id"]
            else:
                db.execute(
                    "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (sample_id, digest, category, str(image), "initial-calibration-reuse", "initialization",
                     "OK", "OK", "initial_calibration_yolo_train_ok", None, now),
                )
                db.execute(
                    "INSERT INTO dataset_split_assignment VALUES (?,?,?,?,?)",
                    (sample_id, category, "OK", "train_locked", now),
                )
                created = True
        # Repair a missing membership/link on resume as well as creating it on
        # the first run, so a partial filesystem failure is recoverable.
        self._copy_to_pool(image, effective_sample_id, category, "confirmed_ok_pool")
        if created:
            append_jsonl(self.events_path, {
                "event": "initial_calibration_ok_locked_for_yolo_train", "at": now,
                "sample_id": sample_id, "category": category, "sha256": digest,
                "source_path": str(image), "split": "train_locked",
            })
        return created

    def apply_review(self, image: Path, record: ReviewRecord) -> None:
        if record.label_source not in {"model_assumed_review", "human_review", "folder_ground_truth", "sampling_pseudo_ok"}:
            raise ValueError("unrecognized label_source")
        with self._connect() as db:
            row = db.execute("SELECT model_decision FROM samples WHERE sample_id=?", (record.sample_id,)).fetchone()
            if not row:
                raise KeyError(record.sample_id)
            db.execute(
                "UPDATE samples SET reviewed_decision=?,label_source=?,mask_path=? WHERE sample_id=?",
                (record.reviewed_decision.value, record.label_source, record.mask_path, record.sample_id),
            )
        if record.reviewed_decision == Decision.OK:
            if record.label_source == "sampling_pseudo_ok":
                self._copy_to_pool(image, record.sample_id, record.category, "sampling_pseudo_ok_pool")
            else:
                self._copy_to_pool(image, record.sample_id, record.category, "confirmed_ok_pool")
            if record.model_decision == Decision.NG:
                self._copy_to_pool(image, record.sample_id, record.category, "historical_false_positive_pool")
        else:
            self._copy_to_pool(image, record.sample_id, record.category, "confirmed_ng_pool")
            if record.model_decision == Decision.OK:
                self._copy_to_pool(image, record.sample_id, record.category, "historical_false_negative_pool")
            external = record.label_source == "folder_ground_truth"
            if self._valid_mask(Path(record.mask_path or ""), image, external):
                self._copy_to_pool(image, record.sample_id, record.category, "labeled_pool")
                mask_dest = self.workspace / "data" / record.category / "labeled_pool" / f"{record.sample_id}.mask.png"
                if external:
                    from detected_pipeline.masks import write_internal_from_external
                    write_internal_from_external(Path(record.mask_path), mask_dest)
                else:
                    from detected_pipeline.masks import internal_mask
                    import cv2
                    mask=internal_mask(Path(record.mask_path)).astype("uint8")*255
                    cv2.imencode(".png",mask)[1].tofile(mask_dest)
            else:
                self._copy_to_pool(image, record.sample_id, record.category, "annotation_qc_failed_pool")
        append_jsonl(self.events_path, {
            "event": "review_applied", "at": utc_now(), "sample_id": record.sample_id,
            "category": record.category, "model_decision": record.model_decision.value,
            "reviewed_decision": record.reviewed_decision.value, "label_source": record.label_source,
            "mask_path": record.mask_path, "reviewer": record.reviewer, "metadata": record.metadata,
        })

    @staticmethod
    def _valid_mask(mask: Path, image: Path, external: bool = True) -> bool:
        if not mask.is_file() or mask.stat().st_size == 0:
            return False
        try:
            from PIL import Image
            import numpy as np
            with Image.open(mask) as mask_image, Image.open(image) as source_image:
                values = np.asarray(mask_image.convert("L"))
                anomaly = values < 128 if external else values >= 128
                return mask_image.size == source_image.size and bool(anomaly.any())
        except Exception:
            return False

    def counts(self, category: str) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT pool,COUNT(*) n FROM pool_membership pm JOIN samples s USING(sample_id) "
                "WHERE s.category=? GROUP BY pool", (category,),
            ).fetchall()
        result = {pool: 0 for pool in POOLS}
        result.update({row["pool"]: row["n"] for row in rows})
        return result

    def state(self, category: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM category_state WHERE category=?", (category,)).fetchone()
        return dict(row)
