# ============================================
# 1. Importing required libraries
# ============================================
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import torch

from ThreeWToolkit.preprocessing import Windowing
from ThreeWToolkit.core.base_preprocessing import WindowingConfig
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig
from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.assessment.model_assess import ModelAssessment
from ThreeWToolkit.core.enums import TaskType, EventPrefixEnum

# Import LSTM model
from ThreeWToolkit.models import LSTMConfig, LSTM
# ============================================

# ============================================
# 2. Load Dataset
# ============================================
dataset_path = Path("../../dataset")

ds_config = ParquetDatasetConfig(
    path=dataset_path,
    clean_data=True,
    target_class=[0, 1, 2],
    event_type=[EventPrefixEnum.REAL]
)
ds = ParquetDataset(ds_config)

print("Dataset loaded!")
print(f"Total events: {len(ds)}")
# ============================================

# ============================================
# 3. Model + Trainer Setup
# ============================================
window_size = 100

lstm_config = LSTMConfig(
    input_size=1,
    hidden_size=64,
    num_layers=1,
    output_size=3,
    random_seed=11,
)

trainer_config = TrainerConfig(
    optimizer="adam",
    criterion="cross_entropy",
    batch_size=32,
    epochs=20,
    learning_rate=0.001,
    seed=11,
    device="cuda" if torch.cuda.is_available() else "cpu",
    config_model=lstm_config,
    cross_validation=False,
    shuffle_train=True,
)

assessment_config = ModelAssessmentConfig(
    metrics=["balanced_accuracy", "precision", "recall", "f1"],
    task_type=TaskType.CLASSIFICATION,
    class_names=["Class_0", "Class_1", "Class_2"],
    export_results=True,
    generate_report=False,
)

trainer = ModelTrainer(trainer_config)
assessor = ModelAssessment(assessment_config)

print("Model Architecture:")
print(trainer.model)
# ============================================

# ============================================
# 4. Windowing
# ============================================
selected_col = "T-TPT"
dfs = []

wind = Windowing(WindowingConfig(
    window="hann",
    window_size=window_size,
    overlap=0.5,
    pad_last_window=True
))

label_wind = Windowing(WindowingConfig(
    window="boxcar",
    window_size=window_size,
    overlap=0.5,
    pad_last_window=True
))

print("Processing events...")
for event in tqdm(ds):
    windowed_signal = wind(event["signal"][selected_col])
    windowed_label = label_wind(event["label"]["class"])

    windowed_signal.drop(columns=["win"], inplace=True)
    windowed_label.drop(columns=["win"], inplace=True)

    windowed_signal["label"] = windowed_label.mode(axis=1)[0].values
    windowed_signal["label"] = windowed_signal["label"].astype(int)

    dfs.append(windowed_signal)

dfs_final = pd.concat(dfs, ignore_index=True)

print("Windowing complete!")
print("Label distribution:")
print(dfs_final["label"].value_counts())
# ============================================

# ============================================
# 5. Model Training
# ============================================
x_train = dfs_final.iloc[:, :-1].values.astype("float32")
y_train = dfs_final["label"].values.astype("int64")

# LSTM expects 3D: (batch, seq_len, features)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

trainer.train(x_train=x_train, y_train=y_train, epochs=trainer_config.epochs)

print("Training completed!")

# ============================================

# ============================================
# 6. Evaluation
# ============================================
results = assessor.evaluate(trainer.model, x_train, y_train)

print("\n=== Evaluation Results ===")
results
