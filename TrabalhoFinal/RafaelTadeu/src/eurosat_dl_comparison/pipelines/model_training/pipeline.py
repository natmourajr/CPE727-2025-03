"""Model training pipelines."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_mlp_cv, train_vgg16_cv, train_resnet50_cv, train_vit_cv


def create_mlp_pipeline(**kwargs) -> Pipeline:
    """Create MLP training pipeline with cross-validation."""
    return pipeline(
        [
            node(
                func=train_mlp_cv,
                inputs=["dev_dataset", "cross_val_table", "parameters"],
                outputs="mlp_cv_results",
                name="train_mlp_cv_node"
            )
        ],
        tags=["training", "mlp", "cv"]
    )


def create_vgg16_pipeline(**kwargs) -> Pipeline:
    """Create VGG16 training pipeline with cross-validation."""
    return pipeline(
        [
            node(
                func=train_vgg16_cv,
                inputs=["dev_dataset", "cross_val_table", "parameters"],
                outputs="vgg16_cv_results",
                name="train_vgg16_cv_node"
            )
        ],
        tags=["training", "vgg16", "cv"]
    )


def create_resnet50_pipeline(**kwargs) -> Pipeline:
    """Create ResNet50 training pipeline with cross-validation."""
    return pipeline(
        [
            node(
                func=train_resnet50_cv,
                inputs=["dev_dataset", "cross_val_table", "parameters"],
                outputs="resnet50_cv_results",
                name="train_resnet50_cv_node"
            )
        ],
        tags=["training", "resnet50", "cv"]
    )


def create_vit_pipeline(**kwargs) -> Pipeline:
    """Create Vision Transformer training pipeline with cross-validation."""
    return pipeline(
        [
            node(
                func=train_vit_cv,
                inputs=["dev_dataset", "cross_val_table", "parameters"],
                outputs="vit_cv_results",
                name="train_vit_cv_node"
            )
        ],
        tags=["training", "vit", "cv"]
    )


def create_all_training_pipeline(**kwargs) -> Pipeline:
    """Create combined training pipeline for all models with cross-validation."""
    return (
        create_mlp_pipeline() +
        create_vgg16_pipeline() +
        create_resnet50_pipeline() +
        create_vit_pipeline()
    )
