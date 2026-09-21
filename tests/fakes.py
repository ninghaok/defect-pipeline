from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from detected_pipeline.contracts import Decision, PretrainedPrediction


class FakePretrainedPlugin:
    def __init__(self, output: Path, decisions: dict[str, tuple[Decision, bool, bool]] | None = None):
        self.output = output
        self.decisions = decisions or {}

    def load(self, config): pass
    def healthcheck(self): return {"ok": True}
    def close(self): pass

    def predict(self, image_path, context):
        decision, boundary, conflict = self.decisions.get(image_path.stem, (Decision.OK, False, False))
        self.output.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            size = image.size
        score = self.output / f"{context.sample_id}.npy"
        mask = self.output / f"{context.sample_id}.png"
        values = np.ones((size[1], size[0]), dtype=np.float32) if decision == Decision.NG else np.zeros((size[1], size[0]), dtype=np.float32)
        np.save(score, values)
        Image.fromarray((values * 255).astype(np.uint8)).save(mask)
        first = Decision.OK if conflict else decision
        return PretrainedPrediction(
            context.sample_id, context.category, float(values.max()), first,
            float(values.max()), decision, decision, boundary, conflict,
            str(score), str(mask), {"feature": .5, "pixel": .5}, 1.0,
            "fake", "1", "fake-1",
        )


def make_image(path: Path, value: int = 128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), value).save(path)
    return path

