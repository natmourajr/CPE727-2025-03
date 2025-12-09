import matplotlib.pyplot as plt

from ThreeWToolkit.pipeline import Pipeline
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_preprocessing import (
    ImputeMissingConfig,
    NormalizeConfig,
    WindowingConfig,
)
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.core.enums import TaskType
from ThreeWToolkit.models.lstm import LSTMConfig
from ThreeWToolkit.trainer.trainer import TrainerConfig

# Define dataset path
dataset_path = "../../dataset"

# -----------------------------------------------------------
# LSTM MODEL CONFIGURATION
# -----------------------------------------------------------
config_model_lstm = LSTMConfig(
    input_size=2,       # number of feature columns being used
    hidden_size=64,     # LSTM hidden units
    num_layers=2,       # number of LSTM stacked layers
    dropout=0.1,
    bidirectional=False,
    output_size=3,      # number of target classes
    random_seed=42,
)

# -----------------------------------------------------------
# PIPELINE DEFINITION
# -----------------------------------------------------------
pipeline_lstm = Pipeline(
    [
        # 1. Load data
        ParquetDatasetConfig(
            path=dataset_path,
            split=None,
            columns=["T-JUS-CKP", "T-MON-CKP"],
            target_column="class",
            target_class=[0, 1, 2],
        ),

        # 2. (Optional but recommended) Missing-value imputation
        ImputeMissingConfig(strategy="mean"),

        # 3. Normalization
        NormalizeConfig(method="zscore"),

        # 4. Windowing for LSTM (sequence length = 200)
        # WindowingConfig(
        #     window_size=200,
        #     horizon=1,          # classification → typically horizon=1
        #     drop_incomplete=True,
        #     reshape_3d = True
        # ),

        # 5. Training
        TrainerConfig(
            optimizer="adam",
            criterion="cross_entropy",
            batch_size=32,
            epochs=20,
            seed=42,
            config_model=config_model_lstm,
            learning_rate=0.001,
            cross_validation=False,
            shuffle_train=True,
        ),

        # 6. Model performance assessment
        ModelAssessmentConfig(
            metrics=["balanced_accuracy", "precision", "recall", "f1"],
            task_type=TaskType.CLASSIFICATION,
            export_results=True,
            generate_report=False,
        ),
    ]
)

print("LSTM Pipeline created successfully!")
print("This pipeline will:")
print("1. Load dataset with 3 classes")
print("2. Impute missing values and normalize data")
print("3. Create windows of size 200 for LSTM input")
print("4. Train an LSTM with 2 layers and 64 hidden units")
print("5. Evaluate using balanced_accuracy, precision, recall, F1")

# -----------------------------------------------------------
# RUN THE PIPELINE
# -----------------------------------------------------------
pipeline_lstm.run()
