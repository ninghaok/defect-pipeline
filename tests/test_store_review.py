import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from detected_pipeline.contracts import Decision, InferenceContext
from detected_pipeline.feedback import FeedbackStore
from detected_pipeline.plugins import PluginLoadError, load_pretrained_plugin
from detected_pipeline.registry import ModelRegistry
from detected_pipeline.review import FolderGroundTruthReviewProvider

from fakes import FakePretrainedPlugin, make_image

CATEGORIES = ["qiumian_fupai", "qiumian_xiepai", "di_mian_detection", "wa_yuan_detection"]


class StoreReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_plugin_fails_closed(self):
        with self.assertRaisesRegex(PluginLoadError, "not configured"):
            load_pretrained_plugin(self.root / "missing.yaml")

    def test_folder_truth_review_routes_fp_fn_and_labeled_pool(self):
        workspace, source = self.root / "workspace", self.root / "dataset" / CATEGORIES[0]
        ok = make_image(source / "test" / "OK" / "false_alarm.png", 10); ng = make_image(source / "test" / "NG" / "miss.png", 20)
        true_mask = source / "mask" / "miss_t.png"; true_mask.parent.mkdir(parents=True, exist_ok=True)
        mask_image = Image.new("L", (16, 12), 255)
        for y in range(4, 8):
            for x in range(6, 10):
                mask_image.putpixel((x, y), 0)
        mask_image.save(true_mask)
        plugin = FakePretrainedPlugin(self.root / "predictions", {ok.stem: (Decision.NG, False, False), ng.stem: (Decision.OK, False, False)})
        reviewer = FolderGroundTruthReviewProvider(); store = FeedbackStore(workspace, CATEGORIES)
        for image in (ok, ng):
            context = InferenceContext(f"{CATEGORIES[0]}-{image.stem}", CATEGORIES[0])
            prediction = plugin.predict(image, context); self.assertTrue(store.ingest(image, context, prediction))
            store.apply_review(image, reviewer.review_prediction(prediction, image))
        counts = store.counts(CATEGORIES[0])
        self.assertEqual(counts["historical_false_positive_pool"], 1); self.assertEqual(counts["historical_false_negative_pool"], 1)
        self.assertEqual(counts["labeled_pool"], 1); self.assertEqual(counts["confirmed_ok_pool"], 1)
        self.assertTrue((workspace / "data" / CATEGORIES[0] / "labeled_pool" / f"{CATEGORIES[0]}-miss.mask.png").is_file())
        self.assertFalse(store.ingest(ok, InferenceContext("dup", CATEGORIES[0]), plugin.predict(ok, InferenceContext("dup", CATEGORIES[0]))))

    def test_registry_promote_and_rollback(self):
        registry = ModelRegistry(self.root / "workspace")
        first = self.root / "one.pt"; first.write_bytes(b"one"); second = self.root / "two.pt"; second.write_bytes(b"two")
        smoke = lambda _: {"ok": True}
        registry.promote(CATEGORIES[0], registry.register_candidate(CATEGORIES[0], "v1", first, "d1", {"image_threshold": .5}, smoke))
        registry.promote(CATEGORIES[0], registry.register_candidate(CATEGORIES[0], "v2", second, "d2", {"image_threshold": .6}, smoke))
        self.assertEqual(registry.current(CATEGORIES[0])["model_version"], "v2")
        self.assertEqual(registry.rollback(CATEGORIES[0])["model_version"], "v1")
        with self.assertRaisesRegex(ValueError, "smoke test failed"):
            registry.register_candidate(CATEGORIES[1], "bad", first, "d", {}, lambda _: {"ok": False})


if __name__ == "__main__":
    unittest.main()
