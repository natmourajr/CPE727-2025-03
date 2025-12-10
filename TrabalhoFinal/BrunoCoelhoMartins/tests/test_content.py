from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.enums import EventPrefixEnum
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("../../dataset")
FOLDS_CSV = DATASET_PATH / "folds/folds_clf_02.csv"

# Load folds
folds = pd.read_csv(FOLDS_CSV)
train_files = folds[folds["fold"].isin([0,1,2])]["instancia"].tolist()
val_files   = folds[folds["fold"]==3]["instancia"].tolist()
test_files  = folds[folds["fold"]==4]["instancia"].tolist()

# Convert to Parquet paths relative to dataset root
train_files = [Path(f) for f in train_files]
val_files   = [Path(f) for f in val_files]
test_files  = [Path(f) for f in test_files]

# ---------- Load separate ParquetDatasets ----------
train_cfg = ParquetDatasetConfig(
    path=DATASET_PATH,
    clean_data=True,
    target_class=None,
    event_type=[EventPrefixEnum.REAL],
    split="list",
    file_list=train_files,
)
ds_train = ParquetDataset(train_cfg)

val_cfg = ParquetDatasetConfig(
    path=DATASET_PATH,
    clean_data=True,
    target_class=None,
    event_type=[EventPrefixEnum.REAL],
    split="list",
    file_list=val_files,
)
ds_val = ParquetDataset(val_cfg)

test_cfg = ParquetDatasetConfig(
    path=DATASET_PATH,
    clean_data=True,
    target_class=None,
    event_type=[EventPrefixEnum.REAL],
    split="list",
    file_list=test_files,
)
ds_test = ParquetDataset(test_cfg)

print(f"Train size: {len(ds_train)}, Val size: {len(ds_val)}, Test size: {len(ds_test)}")
