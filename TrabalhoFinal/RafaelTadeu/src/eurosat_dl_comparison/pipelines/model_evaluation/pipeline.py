"""Model evaluation pipelines."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_best_models_config,
    evaluate_mlp,
    evaluate_vgg16,
    evaluate_resnet50,
    evaluate_vit
)


def create_evaluation_pipeline(**kwargs) -> Pipeline:
    """Create evaluation pipeline for all best models."""
    return pipeline(
        [
            node(
                func=evaluate_mlp,
                inputs=["best_models_config", "dev_dataset", "test_dataset", "parameters"],
                outputs="mlp_evaluation_results",
                name="evaluate_mlp_node"
            ),
            node(
                func=evaluate_vgg16,
                inputs=["best_models_config", "dev_dataset", "test_dataset", "parameters"],
                outputs="vgg16_evaluation_results",
                name="evaluate_vgg16_node"
            ),
            node(
                func=evaluate_resnet50,
                inputs=["best_models_config", "dev_dataset", "test_dataset", "parameters"],
                outputs="resnet50_evaluation_results",
                name="evaluate_resnet50_node"
            ),
            node(
                func=evaluate_vit,
                inputs=["best_models_config", "dev_dataset", "test_dataset", "parameters"],
                outputs="vit_evaluation_results",
                name="evaluate_vit_node"
            )
        ],
        tags=["evaluation"]
    )
