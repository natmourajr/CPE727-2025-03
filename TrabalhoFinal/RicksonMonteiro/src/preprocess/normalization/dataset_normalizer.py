from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List

import unicodedata
import re

from src.preprocess.normalization.class_normalizer import ClassNormalizer
from src.preprocess.normalization.path_normalizer import PathNormalizer


# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------
def clean_text(s: str) -> str:
    """
    Light normalization (idempotent):
        - lowercase
        - remove accents
        - trim

    Does NOT convert to snake_case.
    """
    if not isinstance(s, str):
        return s

    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.strip()


def safe_snake_case(s: str) -> str:
    """
    Converts multi-word strings to snake_case.
    Single-word names remain as-is.

    Examples:
        "fragmento de micro fossíl" → "fragmento_de_micro_fossil"
        "alga"                      → "alga"
    """
    if not isinstance(s, str):
        return s

    s = clean_text(s)

    # Only multiword strings get snake_case
    if " " not in s:
        return s

    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


# =====================================================================
#                     DATASET NORMALIZER (NO GROUPS)
# =====================================================================
class DatasetNormalizer:
    """
    Final canonical dataset normalizer (no group support).

    Responsibilities:
        - Normalize file paths
        - Normalize category names:
              accents → lowercase → (optional aliases) → snake_case
        - Deterministic global category ordering
        - category_id remapping

    Notes:
        - Alias map controls semantic merging.
        - snake_case guarantees canonical formatting.
        - Idempotent: running twice yields the same result.
    """

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        alias_map: Optional[Dict[str, List[str]]] = None,
        lowercase_classes: bool = True,
        strict_classes: bool = False,
        enable_alias_map: Optional[bool] = None,
    ) -> None:

        self.alias_map = alias_map or {}

        # Respect YAML fully
        self.use_alias = (
            bool(self.alias_map) if enable_alias_map is None else enable_alias_map
        )

        # Alias normalizer
        self.class_normalizer = ClassNormalizer(
            alias_map=self.alias_map,
            lowercase=lowercase_classes,
            strict=strict_classes,
        )

        # Path resolver
        self.path_normalizer: Optional[PathNormalizer] = None

        # Optional user-defined category order
        self.predefined_classes = (
            [safe_snake_case(c) for c in classes]
            if classes else None
        )

    # ----------------------------------------------------------------------
    # MAIN EXECUTION
    # ----------------------------------------------------------------------
    def normalize(self, data: Dict[str, Any], root: Path) -> Dict[str, Any]:
        """
        Normalize a harmonized dataset.

        Steps:
            - Ensure category_name exists
            - Normalize file paths
            - Normalize annotation category names
            - Alias first → snake_case later
            - Deterministic categories
            - Remap category_id
        """

        images = data["images"]
        ann_list = data["annotations"]
        categories = data["categories"]

        # --------------------------------------------------------------
        # 0. Ensure category_name exists for each annotation
        # --------------------------------------------------------------
        id_to_name = {c["id"]: c["name"] for c in categories}

        for ann in ann_list:
            if "category_name" not in ann:
                ann["category_name"] = id_to_name[ann["category_id"]]

        # --------------------------------------------------------------
        # 1. Normalize file paths
        # --------------------------------------------------------------
        self.path_normalizer = PathNormalizer(root=root)
        for img in images:
            img["file_name"] = self.path_normalizer.normalize_filename(img["file_name"])

        # --------------------------------------------------------------
        # 2. Normalize annotation category names
        # --------------------------------------------------------------
        for ann in ann_list:
            raw = ann["category_name"]

            # basic cleanup
            canonical = clean_text(raw)

            # alias resolution
            if self.use_alias:
                canonical = clean_text(self.class_normalizer.normalize_name(canonical))

            # final canonical form → snake_case only for multiword
            canonical = safe_snake_case(canonical)

            ann["category_name"] = canonical
            self.class_normalizer.register_class(canonical)

        # --------------------------------------------------------------
        # 3. Build final category list
        # --------------------------------------------------------------
        if self.predefined_classes:
            ordered = self.predefined_classes
        else:
            ordered = sorted({ann["category_name"] for ann in ann_list})

        new_categories = [
            {"id": idx, "name": cname}
            for idx, cname in enumerate(ordered)
        ]

        mapping = {c["name"]: c["id"] for c in new_categories}

        # --------------------------------------------------------------
        # 4. Remap annotation category_id
        # --------------------------------------------------------------
        for ann in ann_list:
            ann["category_id"] = mapping[ann["category_name"]]

        # --------------------------------------------------------------
        # 5. Final canonical dataset (no groups)
        # --------------------------------------------------------------        
        dataset = {
            "images": images,
            "annotations": ann_list,
            "categories": new_categories,
        }

        return self._convert_ids_to_one_based(dataset)

    def _convert_ids_to_one_based(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        # Categories
        for cat in dataset["categories"]:
            cat["id"] += 1

        # Images
        for img in dataset["images"]:
            img["id"] += 1

        # Annotations
        for ann in dataset["annotations"]:
            ann["id"] += 1
            ann["image_id"] += 1
            ann["category_id"] += 1

        return dataset
