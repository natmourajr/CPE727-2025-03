from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from src.dataset.statistics.dataset_statistics import DatasetStatistics


class CrossValidator:
    """
    Gera k-fold cross-validation multilabel estratificado.
    Salva statistics em:
        - canonical.json
        - train.json (de cada fold)
        - val.json (de cada fold)
    """

    def __init__(self, canonical_json: str, run_dir: Path,
                 num_folds: int = 5, seed: int = 42):

        self.canonical_json = Path(canonical_json)
        self.run_dir = Path(run_dir)
        self.num_folds = num_folds
        self.seed = seed

        self.dataset_dir = self.run_dir / "dataset"
        self.folds_dir = self.run_dir / "folds"

    # ------------------------------------------------------------
    def load_dataset(self):
        dataset = json.load(open(self.canonical_json, "r"))

        self.images = dataset["images"]
        self.annotations = dataset["annotations"]
        self.categories = dataset["categories"]
        self.info = dataset.get("info", {})
        self.statistics = dataset.get("statistics", {})
        self.licenses = dataset.get("licenses", {})

        # Build multilabel matrix
        cat_ids = [c["id"] for c in self.categories]
        cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

        Y = np.zeros((len(self.images), len(self.categories)), dtype=int)

        img_index = {img["id"]: i for i, img in enumerate(self.images)}

        for ann in self.annotations:
            row = img_index[ann["image_id"]]
            col = cat_to_idx[ann["category_id"]]
            Y[row, col] = 1

        self.Y = Y
        self.image_ids = [img["id"] for img in self.images]

        print(f"Dataset: {len(self.images)} imagens, {len(self.annotations)} anotações.")

    # ------------------------------------------------------------
    def save_original_dataset(self):
        """Salva canonical.json com statistics."""
        self.dataset_dir.mkdir(parents=True, exist_ok=True)     
        # stats = DatasetStatistics().compute({
        #         "images": self.images,
        #         "annotations": self.annotations,
        #         "categories": self.categories
        #     })

        out_json = self.dataset_dir / "canonical.json"

        json.dump({
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
            "info": self.info,
            "licenses": self.licenses,
            "statistics": self.statistics
        }, open(out_json, "w"), indent=2)

        print(f"canonical.json salvo com statistics → {out_json}")

    # ------------------------------------------------------------
    def save_fold(self, fold_idx, train_ids, val_ids):

        fold_dir = self.folds_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Filter subsets
        train_imgs = [img for img in self.images if img["id"] in train_ids]
        val_imgs   = [img for img in self.images if img["id"] in val_ids]

        train_anns = [ann for ann in self.annotations if ann["image_id"] in train_ids]
        val_anns   = [ann for ann in self.annotations if ann["image_id"] in val_ids]

        # Compute statistics
        train_stats = DatasetStatistics().compute({
            "images": train_imgs,
            "annotations": train_anns,
            "categories": self.categories
        })
        val_stats = DatasetStatistics().compute({
            "images": val_imgs,
            "annotations": val_anns,
            "categories": self.categories
        })

        # Save train.json
        json.dump({
            "images": train_imgs,
            "annotations": train_anns,
            "categories": self.categories,
            "info": self.info,
            "licenses": self.licenses,
            "statistics": train_stats
        }, open(fold_dir / "train.json", "w"), indent=2)

        # Save val.json
        json.dump({
            "images": val_imgs,
            "annotations": val_anns,
            "categories": self.categories,
            "info": self.info,
            "licenses": self.licenses,
            "statistics": val_stats
        }, open(fold_dir / "val.json", "w"), indent=2)

        print(f"Fold {fold_idx:02d} salvo com statistics! → "
              f"{len(train_imgs)} train, {len(val_imgs)} val")

        return fold_dir

    # ------------------------------------------------------------
    def run(self):
        print(f"\nGerando {self.num_folds}-fold cross validation…")

        self.load_dataset()
        self.save_original_dataset()

        self.folds_dir.mkdir(parents=True, exist_ok=True)

        mskf = MultilabelStratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.seed
        )

        fold_dirs = []

        for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(self.image_ids, self.Y), start=1):

            train_ids = [self.image_ids[i] for i in train_idx]
            val_ids   = [self.image_ids[i] for i in val_idx]

            fold_dirs.append(self.save_fold(fold_idx, train_ids, val_ids))

        print("\nCross-validation completo com statistics!\n")

        return fold_dirs
