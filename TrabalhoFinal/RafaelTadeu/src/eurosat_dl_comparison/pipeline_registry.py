"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from kedro.framework.project import find_pipelines

from eurosat_dl_comparison.pipelines import (
    data_preprocessing,
    model_training,
    model_evaluation,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to `Pipeline` objects.
    """

    pipelines = find_pipelines()

    # Data preprocessing pipelines
    pipelines["data_preprocessing"] = data_preprocessing.create_pipeline()
    pipelines["dp_download"] = data_preprocessing.create_download_pipeline()
    pipelines["dp_split"] = data_preprocessing.create_split_pipeline()
    pipelines["dp_cross_validation"] = data_preprocessing.create_cross_validation_pipeline()
    pipelines["dp_cv_fold_loaders"] = data_preprocessing.create_cv_fold_loaders_pipeline()
    pipelines["dp_statistics"] = data_preprocessing.create_statistics_pipeline()

    # Model training pipelines
    pipelines["train_mlp"] = model_training.create_mlp_pipeline()
    pipelines["train_vgg16"] = model_training.create_vgg16_pipeline()
    pipelines["train_resnet50"] = model_training.create_resnet50_pipeline()
    pipelines["train_vit"] = model_training.create_vit_pipeline()
    pipelines["train_all"] = model_training.create_all_training_pipeline()

    # Model evaluation pipelines
    pipelines["evaluation"] = model_evaluation.create_evaluation_pipeline()

    return pipelines
