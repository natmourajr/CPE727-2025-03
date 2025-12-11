from __future__ import annotations
from typing import Dict, Any, List


class StructureValidator:
    """
    Validates the overall structure of the parsed dataset.

    Required top-level fields:
        - images:      list of dicts with keys {id, file_name}
        - annotations: list of dicts with keys {id, image_id, category_id, bbox}
        - categories:  list of dicts with keys {id, name}

    This validator only checks structure and required fields.
    It does NOT check bbox correctness or class consistency
    (those are handled by other modules).
    """

    REQUIRED_TOP_KEYS = ["images", "annotations", "categories"]

    REQUIRED_IMAGE_KEYS = ["id", "file_name"]
    REQUIRED_ANNOT_KEYS = ["id", "image_id", "category_id", "bbox"]
    REQUIRED_CATEGORY_KEYS = ["id", "name"]

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    # ----------------------------------------------------------------------
    # Top-level validation
    # ----------------------------------------------------------------------

    def validate(self, data: Dict[str, Any]) -> None:
        """
        Validate the presence and structure of all required top-level fields.
        Raises ValueError if structure is invalid.
        """

        self._validate_top_keys(data)
        self._validate_images(data["images"])
        self._validate_annotations(data["annotations"])
        self._validate_categories(data["categories"])

    # ----------------------------------------------------------------------
    # Internal checks
    # ----------------------------------------------------------------------

    def _validate_top_keys(self, data: Dict[str, Any]) -> None:
        """Check presence of images, annotations, categories."""
        for key in self.REQUIRED_TOP_KEYS:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in dataset.")
            if not isinstance(data[key], list):
                raise ValueError(f"'{key}' must be a list.")

    def _validate_images(self, images: List[Dict[str, Any]]) -> None:
        """Validate minimal required fields for each image."""
        for img in images:
            for key in self.REQUIRED_IMAGE_KEYS:
                if key not in img:
                    raise ValueError(f"Image entry missing required field '{key}': {img}")

            # file_name must be string-like
            if not isinstance(img["file_name"], str):
                raise ValueError(f"Image file_name must be a string: {img}")

            # id must be int
            if not isinstance(img["id"], int):
                raise ValueError(f"Image id must be an int: {img}")

    def _validate_annotations(self, annotations: List[Dict[str, Any]]) -> None:
        """Validate minimal required fields for each annotation."""
        for ann in annotations:
            for key in self.REQUIRED_ANNOT_KEYS:
                if key not in ann:
                    raise ValueError(f"Annotation missing required field '{key}': {ann}")

            if not isinstance(ann["id"], int):
                raise ValueError(f"Annotation id must be an int: {ann}")

            if not isinstance(ann["image_id"], int):
                raise ValueError(f"Annotation image_id must be an int: {ann}")

            if not isinstance(ann["category_id"], int):
                raise ValueError(f"Annotation category_id must be an int: {ann}")

            bbox = ann["bbox"]
            if not (isinstance(bbox, list) and len(bbox) == 4):
                raise ValueError(f"bbox must be a list of four numbers: {ann}")

    def _validate_categories(self, categories: List[Dict[str, Any]]) -> None:
        """Validate categories: must contain id and name."""
        for cat in categories:
            for key in self.REQUIRED_CATEGORY_KEYS:
                if key not in cat:
                    raise ValueError(f"Category missing required field '{key}': {cat}")

            if not isinstance(cat["id"], int):
                raise ValueError(f"Category id must be an int: {cat}")

            if not isinstance(cat["name"], str):
                raise ValueError(f"Category name must be a string: {cat}")
