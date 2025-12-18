from __future__ import annotations
from pathlib import Path
import json

from typing import Any, Dict, Optional, Union
import yaml  # type: ignore
import pandas as pd  # type: ignore

from .dataset_canonical import DatasetCanonical


class DatasetLoader:
    """
    Load a canonical dataset from disk.

    Supports:
      - json file containing full canonical dict
      - yaml file containing full canonical dict
      - folder produced by parquet saver (images.parquet etc.)

    Returns:
      DatasetCanonical instance.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else Path(".")
        if not self.root.exists():
            raise FileNotFoundError(f"Root path does not exist: {self.root}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Union[str, Path]) -> DatasetCanonical:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")

        if p.is_dir():
            # Expect parquet-style folder with images.parquet etc.
            return self._load_from_folder(p)

        suffix = p.suffix.lower()
        if suffix in (".json",):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DatasetCanonical(data=data, root=self.root)
        if suffix in (".yml", ".yaml"):
            if yaml is None:
                raise RuntimeError("PyYAML required to load YAML files. Install pyyaml.")
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return DatasetCanonical(data=data, root=self.root)
        if suffix in (".parquet",):
            # If user passed a single parquet file, try to read with pandas as table
            if pd is None:
                raise RuntimeError("pandas is required to load parquet. Install pandas + pyarrow.")
            df = pd.read_parquet(p)
            # Heuristic: if contains columns for images -> wrap in dict
            # This is a best-effort: prefer folder-based parquet saver.
            data = {"images": df.to_dict(orient="records")}
            return DatasetCanonical(data=data, root=self.root)

        raise ValueError(f"Unsupported file type: {p.suffix}")

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_from_folder(self, folder: Path) -> DatasetCanonical:
        """
        Load folder produced by DatasetSaver._write_parquet()
        Expect files: images.parquet, annotations.parquet, categories.parquet, metadata.json
        """
        images, annotations, categories = [], [], []

        if pd is None:
            raise RuntimeError(
                "pandas is required to read parquet folder. Install pandas + pyarrow."
            )

        if (folder / "images.parquet").exists():
            df_images = pd.read_parquet(folder / "images.parquet")
            images = df_images.to_dict(orient="records")

        if (folder / "annotations.parquet").exists():
            df_ann = pd.read_parquet(folder / "annotations.parquet")
            annotations = df_ann.to_dict(orient="records")

        if (folder / "categories.parquet").exists():
            df_cat = pd.read_parquet(folder / "categories.parquet")
            categories = df_cat.to_dict(orient="records")

        meta_path = folder / "metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        data: Dict[str, Any] = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        data.update(meta)
        return DatasetCanonical(data=data, root=self.root)
