# ============================================
# minimal_pipeline_multi_models.py
# ============================================

from pathlib import Path
from ThreeWToolkit.pipeline import Pipeline
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_preprocessing import (
    ImputeMissingConfig,
    NormalizeConfig,
    WindowingConfig,
)
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.core.enums import TaskType
from ThreeWToolkit.models import LSTMConfig, CNNConfig, GRUConfig
from ThreeWToolkit.trainer.trainer import TrainerConfig

# ==============================
# Dataset path
# ==============================
dataset_path = Path("../../dataset")

# ==============================
# Windowing config
# ==============================
window_size = 100
window_config = WindowingConfig(window_size=window_size)

# ==============================
# Model configurations
# ==============================
lstm_config = LSTMConfig(
    input_size=2,  # number of features (columns)
    hidden_size=64,
    num_layers=1,
    output_size=3,
    random_seed=42,
)

cnn_config = CNNConfig(
    input_channels=2,
    output_size=3,
    conv_channels=[16, 32],
    kernel_sizes=[3, 3],
    strides=[1, 1],
    pool_kernel_sizes=[2, 2],
    activation_function="relu",
    random_seed=42,
)

gru_config = GRUConfig(
    input_size=2,
    hidden_size=64,
    num_layers=1,
    output_size=3,
    random_seed=42,
)

# ==============================
# Trainer configuration template
# ==============================
def get_trainer_config(model_config):
    return TrainerConfig(
        optimizer="adam",
        criterion="cross_entropy",
        batch_size=32,
        epochs=20,
        seed=42,
        config_model=model_config,
        learning_rate=0.001,
        cross_validation=False,
        shuffle_train=True,
    )

# ==============================
# Model assessment config
# ==============================
assessment_config = ModelAssessmentConfig(
    metrics=["balanced_accuracy", "precision", "recall", "f1"],
    task_type=TaskType.CLASSIFICATION,
    export_results=True,
    generate_report=True,
)

# ==============================
# Preprocessing steps (shared)
# ==============================
preprocessing_steps = [
    ImputeMissingConfig(strategy="median", columns=["T-JUS-CKP", "T-MON-CKP"]),
    NormalizeConfig(norm="l2"),
    window_config,
]

# ==============================
# Pipeline creation
# ==============================
pipelines = {
    "LSTM": Pipeline(
        [
            ParquetDatasetConfig(
                path=dataset_path,
                split=None,
                columns=["T-JUS-CKP", "T-MON-CKP"],
                target_column="class",
                target_class=[0, 1, 2],
            ),
            *preprocessing_steps,
            get_trainer_config(lstm_config),
            assessment_config,
        ]
    ),
    "CNN": Pipeline(
        [
            ParquetDatasetConfig(
                path=dataset_path,
                split=None,
                columns=["T-JUS-CKP", "T-MON-CKP"],
                target_column="class",
                target_class=[0, 1, 2],
            ),
            *preprocessing_steps,
            get_trainer_config(cnn_config),
            assessment_config,
        ]
    ),
    "GRU": Pipeline(
        [
            ParquetDatasetConfig(
                path=dataset_path,
                split=None,
                columns=["T-JUS-CKP", "T-MON-CKP"],
                target_column="class",
                target_class=[0, 1, 2],
            ),
            *preprocessing_steps,
            get_trainer_config(gru_config),
            assessment_config,
        ]
    ),
}

# ==============================
# Run pipelines
# ==============================
for name, pipe in pipelines.items():
    print(f"\n============================")
    print(f"Running pipeline for {name}")
    print(f"============================\n")

    # Run preprocessing and feature extraction to get the data
    dfs_final = pipe.run()  # make sure your pipeline returns the final DataFrame (adjust if needed)

    # Only reshape for RNN/CNN models
    model_type = name  # "LSTM", "GRU", or "CNN"
    if model_type in ["LSTM", "GRU"]:
        # Extract X and Y using the pipeline's holdout
        x_train, y_train, x_test, y_test = pipe.step_model_training.holdout(
            X=dfs_final.iloc[:, :-1],
            Y=dfs_final["label"].astype(int),
            test_size=pipe.step_model_training.test_size,
        )
        # Reshape for RNN: (batch, seq_len, input_size)
        x_train = x_train.values.reshape(-1, window_size, 2)
        x_test = x_test.values.reshape(-1, window_size, 2)

        # Train model
        pipe.step_model_training((x_train, y_train), x_val=x_test, y_val=y_test)
        # Assess
        pipe.step_model_assessment((pipe.step_model_training.model, x_test, y_test))
    else:
        # For MLP or other models that take flattened features
        pipe.run()

    print(f"\n{name} pipeline completed!\n")

