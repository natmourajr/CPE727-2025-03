"""
GroupLabelTransformer

Transforms a canonical dataset into a grouped-label dataset.

Example:
    fossil-related classes  → "fossil"
    fragment subclasses     → "fragment"

Only groups become final categories.
Classes without a group are dropped (optional future parameter).

This transformer:
    - builds new category list using the group_map (as names)
    - replaces annotation category_name → its group_name
    - remaps category_id accordingly
"""

from __future__ import annotations
from typing import Dict, Any, List
from src.preprocess.transformer.base_transformer import BaseTransformer


class GroupLabelTransformer(BaseTransformer):
    """
    Dataset transformer that collapses fine-grained categories into groups.

    Workflow:
        - Read group_map from canonical dataset
        - Build new categories list using only groups
        - Replace annotation labels by their group label
        - Drop annotations whose category is not mapped to any group

    Notes:
        - Images are preserved exactly as canonical
        - Statistics & splits are recomputed automatically
        - Output categories correspond to groups ONLYs
    """

    def __init__(self, group_map: Dict[str, List[str]]):
        """
        Args:
            group_map: dict[str, list[str]]
                Example:
                    {
                        "fossil": ["alga", "gastropoda", "crustacea", ...],
                        "fragment": ["fragmento_de_microfossil"]
                    }
        """
        self.group_map = group_map

        # Precompute reverse lookup for fast grouping:
        #   label → group_name
        self.label_to_group = {}
        for group_name, members in group_map.items():
            for m in members:
                self.label_to_group[m] = group_name

    # ------------------------------------------------------------------
    # 1. Build new grouped categories
    # ------------------------------------------------------------------
    def transform_categories(self, old_categories: List[Dict[str, Any]]):
        """
        Output categories are ONLY the groups.

        Example output:
            [
                { "id": 1, "name": "fossil" },
                { "id": 2, "name": "fragment" }
            ]
        """
        group_names = sorted(self.group_map.keys())

        return [
            {"id": idx, "name": gname}
            for idx, gname in enumerate(group_names, start=1) # COCO format IDs start at 1
        ]

    # ------------------------------------------------------------------
    # 2. Transform annotations to use group labels
    # ------------------------------------------------------------------
    def transform_annotations(
        self,
        old_annotations: List[Dict[str, Any]],
        name_to_id: Dict[str, int],
    ):
        """
        Convert annotation category_name → group_name.

        Any category not in a group is dropped.
        """

        new_annotations = []
        next_ann_id = 1

        for ann in old_annotations:
            cname = ann["category_name"]

            # Only keep annotations mapped to some group
            group_name = self.label_to_group.get(cname)
            if group_name is None:
                continue  # drop non-grouped categories

            new_ann = ann.copy()
            new_ann["id"] = next_ann_id
            new_ann["category_id"] = name_to_id[group_name]
            new_ann["category_name"] = group_name

            new_annotations.append(new_ann)
            next_ann_id += 1

        return new_annotations
