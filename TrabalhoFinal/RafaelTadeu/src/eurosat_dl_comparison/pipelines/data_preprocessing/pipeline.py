"""Data preprocessing pipeline."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_test_loader,
    download_eurosat,
    split_dataset,
    create_cross_validation_dataset,
    create_cv_fold_loaders,
    compute_dataset_statistics,
)


def create_download_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for downloading EuroSAT dataset.

    Returns:
        Pipeline for dataset download.
    """
    return pipeline(
        [
            node(
                func=download_eurosat,
                inputs="params:dataset",
                outputs="raw_dataset",
                name="download_eurosat_node",
            ),
        ]
    )


def create_split_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for splitting dataset into dev and test sets.

    Returns:
        Pipeline for dataset splitting.
    """
    return pipeline(
        [
            node(
                func=split_dataset,
                inputs=["raw_dataset", "params:train_test_split"],
                outputs=["dev_dataset", "test_dataset"],
                name="split_dataset_node",
            ),
        ]
    )


def create_cross_validation_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for generating cross-validation folds.

    Returns:
        Pipeline for cross-validation dataset creation.
    """
    return pipeline(
        [
            node(
                func=create_cross_validation_dataset,
                inputs=["dev_dataset", "params:cross_validation"],
                outputs="cross_val_table",
                name="create_cross_validation_node",
            ),
        ]
    )


def create_cv_fold_loaders_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for generating cross-validation fold loaders.

    Note: This pipeline uses a default fold_num parameter. For actual use,
    you'll need to call create_cv_fold_loaders directly with specific fold numbers.

    Returns:
        Pipeline for CV fold loader creation.
    """
    return pipeline(
        [
            node(
                func=create_cv_fold_loaders,
                inputs=[
                    "dev_dataset",
                    "test_dataset",
                    "cross_val_table",
                    "params:data_loaders",
                ],
                outputs=["cv_train_loader", "cv_val_loader", "cv_test_loader"],
                name="create_cv_fold_loaders_node",
            ),
        ]
    )

def create_test_loader_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for generating test loader.

    Returns:
        Pipeline for test loader creation.
    """
    return pipeline(
        [
            node(
                func=create_test_loader,
                inputs=[
                    "test_dataset",
                    "params:data_loaders",
                ],
                outputs=["test_loader"],
                name="create_test_loader_node",
            ),
        ]
    )


def create_statistics_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for computing dataset statistics.

    Returns:
        Pipeline for statistics computation.
    """
    return pipeline(
        [
            node(
                func=compute_dataset_statistics,
                inputs="dev_dataset",
                outputs="dataset_statistics",
                name="compute_statistics_node",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create the complete data preprocessing pipeline.

    Combines all individual pipelines into a single preprocessing pipeline.
    Note: Does not include CV fold loaders as they require fold_num parameter.

    Returns:
        Complete data preprocessing pipeline.
    """
    download_pipeline = create_download_pipeline()
    split_pipeline = create_split_pipeline()
    cross_val_pipeline = create_cross_validation_pipeline()
    statistics_pipeline = create_statistics_pipeline()

    return (
        download_pipeline
        + split_pipeline
        + cross_val_pipeline
        + statistics_pipeline
    )
