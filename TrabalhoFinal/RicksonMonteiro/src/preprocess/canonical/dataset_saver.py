from __future__ import annotations
from pathlib import Path
import json
import shutil
from typing import Any, Dict, Optional, Union, List

import yaml
import pandas as pd

from .dataset_canonical import DatasetCanonical


class DatasetSaver:
    """
    Saves:
        - dataset.json
        - images/canonical/<split>/...
        - images/grouped/<split>/...
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else Path(".")
        self.root.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def save(
        self,
        canonical: Union[DatasetCanonical, Dict[str, Any]],
        name: str,
        formats: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Path]:

        formats = formats or ["json"]

        data = canonical.to_dict() if isinstance(canonical, DatasetCanonical) else canonical
        saved_paths: Dict[str, Path] = {}

        # detect grouped dataset
        is_grouped = name.endswith("_grouped")
        namespace = "grouped" if is_grouped else "canonical"

        # IMAGE OUTPUT ROOT
        images_root = self.root / "images" / namespace
        self._save_images(data, images_root, overwrite)

        # SAVE CANONICAL DATASET (json/yaml/parquet)
        for fmt in formats:
            fmt = fmt.lower()

            if fmt == "json":
                p = self._write_json(data, name, overwrite)

            elif fmt in ("yaml", "yml"):
                p = self._write_yaml(data, name, overwrite)

            elif fmt == "parquet":
                p = self._write_parquet_bundle(data, name, overwrite)

            else:
                raise ValueError(f"Unsupported format: {fmt}")

            saved_paths[fmt] = p

        return saved_paths

    def _save_images(self, data: Dict[str, Any], img_root: Path, overwrite: bool):

        ingested_root = Path("data/interim/ingested")

        # build map: image_id → split
        imgid2split = {}
        for split_name, id_list in data.get("splits", {}).items():
            for img_id in id_list:
                imgid2split[img_id] = split_name

        for img in data["images"]:
            img_id = img["id"]
            filename = img["file_name"]
            exp_id = img["experiment_id"]

            # split = imgid2split[img_id]

            src = ingested_root / exp_id / "images" / filename
            if not src.exists():
                raise FileNotFoundError(f"Missing source image: {src}")

            # dst = img_root / split / filename
            dst = img_root / filename
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                dst.unlink()

            shutil.copy2(src, dst)

    # ==========================================================
    # WRITERS
    # ==========================================================
    def _write_json(self, data, name, overwrite):
        path = self.root / f"{name}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists. Use overwrite=True.")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def _write_yaml(self, data, name, overwrite):
        path = self.root / f"{name}.yaml"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists.")

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return path

    def _write_parquet_bundle(self, data, name, overwrite):
        folder = self.root / name
        if folder.exists() and not overwrite:
            raise FileExistsError(f"{folder} exists.")

        folder.mkdir(parents=True, exist_ok=True)

        if "images" in data:
            pd.DataFrame(data["images"]).to_parquet(folder / "images.parquet", index=False)
        if "annotations" in data:
            pd.DataFrame(data["annotations"]).to_parquet(folder / "annotations.parquet", index=False)
        if "categories" in data:
            pd.DataFrame(data["categories"]).to_parquet(folder / "categories.parquet", index=False)

        meta = {k: v for k, v in data.items() if k not in ("images", "annotations", "categories")}
        with open(folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return folder
