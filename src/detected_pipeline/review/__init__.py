from .assumed import ModelAssumedReviewProvider
from .folder_truth import FolderGroundTruthReviewProvider
from .sampling import plan_review, run_sampled_review


def build_review_provider(config):
    name = config.get("review_provider", "model_assumed_review")
    if name == "model_assumed_review":
        return ModelAssumedReviewProvider()
    if name == "folder_ground_truth":
        return FolderGroundTruthReviewProvider(config.get("folder_ground_truth", {}))
    raise ValueError(f"unknown review_provider: {name}")


__all__ = ["ModelAssumedReviewProvider", "FolderGroundTruthReviewProvider", "build_review_provider", "plan_review", "run_sampled_review"]
