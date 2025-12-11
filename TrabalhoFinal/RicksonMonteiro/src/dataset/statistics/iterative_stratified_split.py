from __future__ import annotations
from typing import Dict, Any, List, Tuple

from iterstrat.ml_stratifiers import IterativeStratification
import numpy as np


class IterativeStratifiedSplit:
    """
    Multi-label stratified splitting using SOTA iterative stratification.

    Responsibilities:
        - Build multi-label matrix for each image
        - Use iterative stratification to preserve label distribution
        - Return train/val/test splits

    Notes:
        - Works for multi-label images (COCO detection)
        - Output splits contain image IDs
        - Does NOT modify dataset; only returns ID lists
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        # ----------------------------------------
        # Validate ratios
        # ----------------------------------------
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio = "
                f"{train_ratio + val_ratio + test_ratio} (must be 1.0)"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def split(
        self,
        images: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
    ) -> Dict[str, List[int]]:
        """
        Returns:
            {
                "train": [...],
                "val":   [...],
                "test":  [...]
            }
        """

        image_ids = [img["id"] for img in images]

        # --------------------------------------------------------------
        # 1. Multi-label matrix Y (images × categories)
        # --------------------------------------------------------------
        cat_ids = [c["id"] for c in categories]
        cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

        Y = self._build_multilabel_matrix(
            image_ids=image_ids,
            annotations=annotations,
            cat_to_idx=cat_to_idx
        )

        # --------------------------------------------------------------
        # 2. First split: train / (val+test)
        # --------------------------------------------------------------
        train_idx, rem_idx = self._iterative_split(Y, self.train_ratio)

        # --------------------------------------------------------------
        # 3. Second split inside remaining: val / test
        # --------------------------------------------------------------
        val_ratio_adj = self.val_ratio / (self.val_ratio + self.test_ratio)
        Y_rem = Y[rem_idx]

        val_idx_rel, test_idx_rel = self._iterative_split(Y_rem, val_ratio_adj)

        val_idx = [rem_idx[i] for i in val_idx_rel]
        test_idx = [rem_idx[i] for i in test_idx_rel]

        # Convert indices → image IDs
        return {
            "train": [image_ids[i] for i in train_idx],
            "val":   [image_ids[i] for i in val_idx],
            "test":  [image_ids[i] for i in test_idx],
        }

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================
    def _build_multilabel_matrix(
        self,
        image_ids: List[int],
        annotations: List[Dict[str, Any]],
        cat_to_idx: Dict[int, int],
    ) -> np.ndarray:
        """
        Create matrix Y[i, j] = 1 if image i contains category j
        """
        n_imgs = len(image_ids)
        n_cats = len(cat_to_idx)

        Y = np.zeros((n_imgs, n_cats), dtype=int)

        img_to_row = {img_id: i for i, img_id in enumerate(image_ids)}

        for ann in annotations:
            img_id = ann["image_id"]
            cid = ann["category_id"]

            if cid not in cat_to_idx:
                continue

            row = img_to_row[img_id]
            col = cat_to_idx[cid]

            Y[row, col] = 1

        return Y

    def _iterative_split(
        self,
        Y: np.ndarray,
        ratio: float,
    ) -> Tuple[List[int], List[int]]:
        """
        2-way iterative stratified split.
        Returns:
            left_indices, right_indices
        """
        splitter = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[ratio, 1 - ratio],
        )

        train_idx, test_idx = next(splitter.split(Y, Y))

        return list(train_idx), list(test_idx)
