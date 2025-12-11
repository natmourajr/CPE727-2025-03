from ThreeWToolkit.pipeline import Pipeline
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_preprocessing import ImputeMissingConfig, NormalizeConfig, WindowingConfig
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.core.enums import TaskType

dataset_path = "../../dataset"

config_model = MLPConfig(
    hidden_sizes=(64, 32),
    output_size=2,
    random_seed=42,
    activation_function="relu",
    regularization=None,
)

pipeline = Pipeline(
    [
        ParquetDatasetConfig(
            path=dataset_path,
            split=None,
            force_download=False,
            columns=["T-JUS-CKP", "T-MON-CKP"],
            target_column="class",
            target_class=[0, 1],
        ),
        ImputeMissingConfig(strategy="median", columns=["T-JUS-CKP"]),
        NormalizeConfig(norm="l2"),
        WindowingConfig(window_size=100),
        TrainerConfig(
            optimizer="adam",
            criterion="cross_entropy",
            batch_size=32,
            epochs=10,
            seed=42,
            config_model=config_model,
            learning_rate=0.001,
            cross_validation=False,
            shuffle_train=True,
        ),
        ModelAssessmentConfig(
            metrics=["balanced_accuracy", "precision", "recall", "f1"],
            task_type=TaskType.CLASSIFICATION,
            export_results=True,
            generate_report=True,
        ),
    ]
)

pipeline.run()
