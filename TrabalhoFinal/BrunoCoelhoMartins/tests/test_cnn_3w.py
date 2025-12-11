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
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.assessment.model_assess import ModelAssessment
from ThreeWToolkit.core.enums import TaskType, EventPrefixEnum

# Import new model
from ThreeWToolkit.models import CNNConfig, CNN
# ============================================

# ============================================
# 2. Loading the 3W Dataset
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

cnn_config = CNNConfig(
    input_size=window_size,
    num_classes=3,
    conv_channels=[16, 32],
    kernel_sizes=[3, 3],
    output_size=64,
    random_seed=11
)

trainer_config = TrainerConfig(
    optimizer="adam",
    criterion="cross_entropy",
    batch_size=32,
    epochs=20,
    learning_rate=0.001,
    seed=11,
    device="cuda" if torch.cuda.is_available() else "cpu",
    config_model=cnn_config,
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
x_train = dfs_final.iloc[:, :-1]
y_train = dfs_final["label"]

trainer.train(x_train=x_train, y_train=y_train)

print("Training completed!")
# ============================================

# ============================================
# 6. Evaluation
# ============================================
results = assessor.evaluate(trainer.model, x_train, y_train)

print("\n=== Evaluation Results ===")
print(results)
