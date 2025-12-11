# ============================================
# 1. Importing required libraries
# ============================================
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.enums import EventPrefixEnum

# ============================================
# 2. Load Dataset
# ============================================
dataset_path = Path("../../dataset")

ds_config = ParquetDatasetConfig(
    path=dataset_path,
    clean_data=True,
    event_type=[EventPrefixEnum.REAL]
)

ds = ParquetDataset(ds_config)

print("Dataset loaded!")
print(f"Total events: {len(ds)}\n")

# ============================================
# 3. Collect ALL labels from the dataset
# ============================================
print("Collecting labels from all events...\n")

all_labels = []

for event in tqdm(ds, desc="Reading labels"):
    labels = event["label"]["class"]    # column with original labels (not windowed!)
    all_labels.extend(labels)

all_labels = np.array(all_labels)

# ============================================
# 4. Print results
# ============================================
unique_labels, counts = np.unique(all_labels, return_counts=True)
percentages = (counts / counts.sum()) * 100

print("\n============================================")
print("         LABEL SUMMARY (3W Dataset)")
print("============================================")
for label, count, pct in zip(unique_labels, counts, percentages):
    print(f"Label {label:>2} → Count: {count:>8}  ({pct:6.2f}%)")

print("\nUnique label values found:", unique_labels)
print(f"Total labeled samples: {len(all_labels)}")
print("============================================\n")
