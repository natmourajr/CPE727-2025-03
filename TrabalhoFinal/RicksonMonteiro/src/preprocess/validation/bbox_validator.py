# src/dataset/validation/bbox_validator.py
from __future__ import annotations
from typing import Dict, Any, List


class BBoxValidator:
    """
    Validate YOLO-style bounding boxes in normalized xywh format.

    Validation rules:
        - bbox must have 4 elements
        - all values must be numeric
        - width and height must be > 0
        - values must be within [0, 1] (soft constraint)
        - x + w <= 1 and y + h <= 1 (soft constraint)
    
    Error severity:
        - MINOR: small out-of-bound errors (< 5%) → fixable
        - MAJOR: bbox partially outside image → clampable
        - CRITICAL: invalid geometry or totally outside → remove
    """

    def __init__(self, strict: bool = False):
        self.strict = strict

    # ------------------------------------------------------------------
    # MAIN VALIDATION
    # ------------------------------------------------------------------
    def validate(self, dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        errors = []

        for ann in dataset.get("annotations", []):
            bbox = ann.get("bbox")

            if bbox is None:
                errors.append(self._err(ann, "missing_bbox"))
                continue

            if len(bbox) != 4:
                errors.append(self._err(ann, "invalid_length"))
                continue

            try:
                x, y, w, h = map(float, bbox)
            except Exception:
                errors.append(self._err(ann, "non_numeric"))
                continue

            # Geometry
            if w <= 0 or h <= 0:
                errors.append(self._err(ann, "critical_nonpositive_area", severity="critical"))
                continue

            # Range issues
            out_of_bounds = (
                x < 0 or y < 0 or w < 0 or h < 0 or
                x > 1 or y > 1 or w > 1 or h > 1 or
                x + w > 1 or y + h > 1
            )

            if out_of_bounds:
                severity = self._classify_severity(x, y, w, h)
                errors.append(self._err(ann, f"{severity}_out_of_bounds", severity=severity))

        return errors

    # ------------------------------------------------------------------
    # SEVERITY CLASSIFICATION
    # ------------------------------------------------------------------
    def _classify_severity(self, x: float, y: float, w: float, h: float) -> str:
        """
        Classify severity of out-of-bounds error.
        """
        overflow_x = max(0, x + w - 1)
        overflow_y = max(0, y + h - 1)
        underflow_x = max(0, -x)
        underflow_y = max(0, -y)

        max_err = max(overflow_x, overflow_y, underflow_x, underflow_y)

        if max_err < 0.05:
            return "minor"
        if max_err < 0.20:
            return "major"
        return "critical"

    # ------------------------------------------------------------------
    def _err(self, ann: Dict[str, Any], msg: str, severity: str = "minor") -> Dict[str, Any]:
        return {
            "annotation_id": ann.get("id"),
            "image_id": ann.get("image_id"),
            "bbox": ann.get("bbox"),
            "error": msg,
            "severity": severity,
        }
