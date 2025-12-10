# src/dataset/transformers/base_transformer.py

from __future__ import annotations
from typing import Dict, Any, List
from copy import deepcopy

from src.dataset.statistics.dataset_statistics import DatasetStatistics
# from src.dataset.splitting.split_strategy_factory import SplitStrategyFactory
from src.preprocess.canonical.dataset_canonical import DatasetCanonical


class BaseTransformer:
    """
    Abstract base class for dataset label transformers.

    Child classes must override:
        - transform_categories(old_categories)
        - transform_annotations(old_annotations, name_to_id)

    The BaseTransformer handles:
        - Extracting & cloning data
        - Rebuilding categories
        - Reassigning IDs
        - Recomputing statistics
        - Recomputing splits
        - Returning a NEW DatasetCanonical
    """

    # ------------------------------------------------------------------
    # MAIN ENTRYPOINT
    # ------------------------------------------------------------------
    def apply(self, dataset: DatasetCanonical | Dict[str, Any]) -> DatasetCanonical:
        """Apply the transformation and return a NEW DatasetCanonical dataset."""

        # --------------------------------------------------------------
        # Extract raw data and root
        # --------------------------------------------------------------
        if isinstance(dataset, DatasetCanonical):
            data = deepcopy(dataset._data)
            root = dataset.root
        else:
            data = deepcopy(dataset)
            root = None  # may be None, DatasetCanonical will handle that

        old_categories = data.get("categories", [])
        old_annotations = data.get("annotations", [])   
        images = data.get("images", [])
        info = data.get("info", {})
        licenses = data.get("licenses", {})

        # --------------------------------------------------------------
        # 1. Transform categories (subclass must implement)
        # --------------------------------------------------------------
        new_categories = self.transform_categories(old_categories)
        if not isinstance(new_categories, list):
            raise TypeError("transform_categories() must return a list")

        # Build a lookup table: name → new_id
        name_to_id = {c["name"]: c["id"] for c in new_categories}

        # --------------------------------------------------------------
        # 2. Transform annotations (subclass must implement)
        # --------------------------------------------------------------
        new_annotations = self.transform_annotations(
            old_annotations,
            name_to_id=name_to_id
        )
        if not isinstance(new_annotations, list):
            raise TypeError("transform_annotations() must return a list")

        # --------------------------------------------------------------
        # 3. Recompute statistics
        # --------------------------------------------------------------
        stats = DatasetStatistics().compute({
            "images": images,
            "annotations": new_annotations,
            "categories": new_categories
        })

        # --------------------------------------------------------------
        # 4. Recompute splits
        # --------------------------------------------------------------
        # splitter = SplitStrategyFactory.create({})
        # splits = splitter.split(images, new_annotations, new_categories)

        # --------------------------------------------------------------
        # 5. Build final transformed dataset
        # --------------------------------------------------------------
        transformed = {
            "images": images,
            "annotations": new_annotations,
            "categories": new_categories,
            "statistics": stats,
            # "splits": splits,
            "info": info,
            "licenses": licenses,
        }

        # --------------------------------------------------------------
        # 6. Return new DatasetCanonical with preserved root
        # --------------------------------------------------------------
        return DatasetCanonical(transformed, root=root)

    # ------------------------------------------------------------------
    # ABSTRACT METHODS
    # ------------------------------------------------------------------
    def transform_categories(self, old_categories: List[Dict[str, Any]]):
        raise NotImplementedError

    def transform_annotations(
        self,
        old_annotations: List[Dict[str, Any]],
        name_to_id: Dict[str, int],
    ):
        raise NotImplementedError
