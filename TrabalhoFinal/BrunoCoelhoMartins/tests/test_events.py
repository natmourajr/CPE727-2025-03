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

# ============================================

# ============================================
# 2. Load Dataset
# ============================================
dataset_path = Path("../../dataset")

ds_config = ParquetDatasetConfig(
    path=dataset_path,
    clean_data=True,
    #target_class=[0, 1, 2]
    event_type=[EventPrefixEnum.REAL]
)
ds = ParquetDataset(ds_config)

print("Dataset loaded!")
print(f"Total events: {len(ds)}")

window_size = 100


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
    windowed_signal = wind(event["signal"])
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