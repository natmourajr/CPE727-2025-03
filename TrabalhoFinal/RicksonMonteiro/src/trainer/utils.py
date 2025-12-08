import json
from typing import Any, Dict
import numpy as np
import torch
from pathlib import Path
import yaml
import random, os

def collate_fn(batch):
    return list(zip(*batch))


def set_seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(fold_dir, history: Dict[str, Any]):
    json.dump(history, open(fold_dir / "metrics.json", "w"), indent=2)


def load_yaml(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg