from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import json


class DatasetCanonical:
    """
    Canonical dataset representation after all cleaning, validation,
    normalization and statistics computation.

    Responsibilities:
        - Hold the final dataset structure
        - Provide easy accessors for images, annotations and categories
        - Provide to_dict() / save_json() utilities
        - Ensure immutability of the canonical dataset (treat as read-only)

    Final Data Format (COCO-like canonical structure):

        {
            "images": [
                {
                    "id": int,
                    "file_name": str,
                    "width": int,
                    "height": int
                }
            ],

            "annotations": [
                {
                    "id": int,
                    "image_id": int,
                    "category_id": int,
                    "bbox": [x, y, w, h],
                    "area": float,
                    "iscrowd": int
                }
            ],

            "categories": [
                {
                    "id": int,
                    "name": str
                }
            ],

            "groups": {                    # Optional (class grouping)
                "<class_name>": {
                    "group": str or None,
                    "group_id": int or None
                }
            },

            "statistics": {                # Optional (dataset statistics)
                "images": {...},
                "categories": {...},
                "bboxes": {...},
                "cooccurrence": {...}
            },

            "splits": {                    # Optional (train/val/test or CV)
                "train": [image_ids],
                "val": [image_ids],
                "test": [image_ids],
                "folds": { ... }           # If k-fold CV was applied
            }
        }

    Notes:
        - All paths must be normalized and relative to dataset root
        - Category IDs must be deterministic and contiguous (0..N-1)
        - DatasetCanonical objects are treated as immutable final artifacts
    """
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    
    def __init__(self, data: Dict[str, Any], root: Path) -> None:
        self._data = data            # internal, treated as immutable
        self.root = Path(root)

        self._validate_canonical_format()

    # ------------------------------------------------------------------
    # Validation (structural, lightweight)
    # ------------------------------------------------------------------

    def _validate_canonical_format(self) -> None:
        """Ensures minimal canonical structure exists and is consistent."""

        required = ["images", "annotations", "categories"]

        for key in required:
            if key not in self._data:
                raise ValueError(
                    f"Canonical dataset missing required field '{key}'. "
                    f"Available keys: {list(self._data.keys())}"
                )

        if not isinstance(self._data["images"], list):
            raise ValueError("'images' must be a list.")

        if not isinstance(self._data["annotations"], list):
            raise ValueError("'annotations' must be a list.")

        if not isinstance(self._data["categories"], list):
            raise ValueError("'categories' must be a list.")

        # Ensure category IDs are deterministic and unique
        # cat_ids = [c["id"] for c in self._data["categories"]]
        # if sorted(cat_ids) != list(range(len(cat_ids))):
        #     raise ValueError(
        #         "Categories must have deterministic IDs: 1..N "
        #         f"(found {cat_ids})"
        #     )

        # Ensure annotation category_id references exist
        cat_ids = [c["id"] for c in self._data["categories"]]
        valid_cat_ids = set(cat_ids)
        for ann in self._data["annotations"]:
            cid = ann["category_id"]
            if cid not in valid_cat_ids:
                raise ValueError(
                    f"Annotation refers to unknown category_id={cid}"
                )

    # ------------------------------------------------------------------
    # Accessors (read-only)
    # ------------------------------------------------------------------

    @property
    def images(self):
        return self._data["images"]

    @property
    def annotations(self):
        return self._data["annotations"]

    @property
    def categories(self):
        return self._data["categories"]

    @property
    def statistics(self) -> Optional[Dict[str, Any]]:
        return self._data.get("statistics")

    @property
    def groups(self) -> Optional[Dict[str, Any]]:
        return self._data.get("groups")

    @property
    def splits(self) -> Optional[Dict[str, Any]]:
        return self._data.get("splits")

    @property
    def info(self) -> Optional[Dict[str, Any]]:
        return self._data.get("info")
    
    @property
    def info(self) -> Optional[Dict[str, Any]]:
        return self._data.get("license")
    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the canonical data."""
        return dict(self._data)

    def save_json(self, path: Path) -> None:
        """
        Save canonical dataset to disk as JSON.
        This is the main format used for:
            - DVC versioning
            - MLflow artifacts
            - dataset interchange between pipelines
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                self._data,
                f,
                indent=2,
                ensure_ascii=False   # preserve Unicode labels / metadata
            )

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        summary = {
            "images": len(self.images),
            "annotations": len(self.annotations),
            "categories": len(self.categories),
            'info': self.info,
            "has_statistics": self.statistics is not None,
            "has_groups": self.groups is not None,
            "has_splits": self.splits is not None,
        }
        return f"DatasetCanonical({summary})"

    def __repr__(self):
        return self.__str__()
