from ThreeWToolkit.pipeline import Pipeline
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_preprocessing import ImputeMissingConfig, NormalizeConfig, WindowingConfig
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.core.enums import TaskType

dataset_path = "../../dataset"

config_model_stats = MLPConfig(
    hidden_sizes=(128, 64, 32),
    output_size=3,
    random_seed=42,
    activation_function="relu",
    regularization=None,
)

pipeline_stats = Pipeline(
    [
        ParquetDatasetConfig(
            path=dataset_path,
            split=None,
            force_download=False,
            columns=["T-JUS-CKP", "T-MON-CKP", "P-PDG"],
            target_column="class",
            target_class=[0, 1, 2],
        ),
        ImputeMissingConfig(strategy="mean", columns=None),  # Impute all columns
        NormalizeConfig(norm="l1"),
        WindowingConfig(window_size=150),
        StatisticalConfig(),  # Extract statistical features
        TrainerConfig(
            optimizer="adam",
            criterion="cross_entropy",
            batch_size=64,
            epochs=50,
            seed=42,
            config_model=config_model_stats,
            learning_rate=0.0005,
            cross_validation=True,
            shuffle_train=True,
        ),
        ModelAssessmentConfig(
            metrics=["balanced_accuracy", "precision", "recall", "f1"],
            task_type=TaskType.CLASSIFICATION,
            export_results=True,
            generate_report=False,
        ),
    ]
)

pipeline_stats.run()
