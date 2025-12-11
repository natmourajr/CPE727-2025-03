from __future__ import annotations
from typing import Dict, Any, List

# Specialized statistics modules
from src.dataset.statistics.class_distribution import ClassDistribution
from src.dataset.statistics.bbox_distribution import BBoxDistribution
from src.dataset.statistics.cooccurrence_matrix import CooccurrenceMatrix


class DatasetStatistics:
    """
    Orchestrates all dataset-level statistics computations.

    Responsibilities:
        - Image statistics (resolution, count)
        - Category distribution (counts, percentages, imbalance, entropy)
        - Bounding box statistics (size, area, ratios)
        - BBoxes per image (count distribution)
        - Class co-occurrence matrix

    NOTE:
        This module ONLY orchestrates and aggregates results.
        Actual computation is delegated to specialized modules.
    """

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def compute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        return {
            "images": self._compute_image_stats(images),

            "categories": ClassDistribution().compute(
                annotations,
                categories
            ),

            "bboxes": BBoxDistribution().compute(
                annotations
            ),

            "bboxes_per_image": self._compute_bboxes_per_image(
                images,
                annotations
            ),

            "cooccurrence": CooccurrenceMatrix().compute(
                images,
                annotations,
                categories
            ),
        }

    # ----------------------------------------------------------------------
    # Image Stats (simple enough to keep internal)
    # ----------------------------------------------------------------------

    def _compute_image_stats(
            self,
            images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not images:
            return {"count": 0}

        import numpy as np

        widths = np.array([img["width"] for img in images], dtype=float)
        heights = np.array([img["height"] for img in images], dtype=float)

        def safe_std(arr):
            return float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        return {
            "count": len(images),

            "width_mean": float(widths.mean()),
            "width_std": safe_std(widths),
            "width_min": float(widths.min()),
            "width_max": float(widths.max()),

            "height_mean": float(heights.mean()),
            "height_std": safe_std(heights),
            "height_min": float(heights.min()),
            "height_max": float(heights.max()),
        }

    # ----------------------------------------------------------------------
    # BBoxes per Image (unique logic → kept internal)
    # ----------------------------------------------------------------------

    def _compute_bboxes_per_image(
        self,
        images: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not images:
            return {"count": 0}

        import numpy as np
        from collections import Counter

        counts = Counter(ann["image_id"] for ann in annotations)

        # Ensure presence of images with zero boxes
        values = np.array([
            counts.get(img["id"], 0)
            for img in images
        ], dtype=float)

        def safe_std(arr):
            return float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        return {
            "mean": float(values.mean()),
            "std": safe_std(values),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
        }
