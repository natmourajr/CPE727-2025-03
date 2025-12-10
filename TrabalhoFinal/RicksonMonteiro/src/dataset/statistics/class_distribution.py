from __future__ import annotations
from typing import Dict, Any, List
from collections import Counter
import numpy as np


class ClassDistribution:
    """
    Compute statistical properties of class distribution.

    Responsibilities:
        - Count instances per category (ID → count)
        - Map to category names
        - Compute imbalance ratio
        - Compute percentage distribution
        - Compute mean/std over class counts
        - Compute entropy of the distribution
        - Provide summary for downstream DatasetStatistics
    """

    def compute(
        self,
        annotations: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        # ------------------------------------------------------------------
        # Count occurrences of each category_id
        # ------------------------------------------------------------------
        counts = Counter(ann["category_id"] for ann in annotations)

        # ------------------------------------------------------------------
        # Build name → count mapping (readable, COCO-like)
        # ------------------------------------------------------------------
        dist = {}
        for cat in categories:
            name = cat["name"]
            cid = cat["id"]
            dist[name] = int(counts.get(cid, 0))

        total = sum(dist.values()) or 1

        # ------------------------------------------------------------------
        # Imbalance ratio = max_nonzero / min_nonzero
        # (ignores classes with zero counts)
        # ------------------------------------------------------------------
        nonzero = [v for v in dist.values() if v > 0]
        imbalance = (max(nonzero) / min(nonzero)) if nonzero else 1.0

        # ------------------------------------------------------------------
        # Mean / Std of class counts
        # ------------------------------------------------------------------
        values = np.array(list(dist.values()), dtype=float)

        def safe_std(arr):
            return float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        mean_count = float(values.mean())
        std_count = safe_std(values)

        # ------------------------------------------------------------------
        # Percentage distribution
        # ------------------------------------------------------------------
        percent_dist = {name: (count / total) for name, count in dist.items()}

        # ------------------------------------------------------------------
        # Entropy of the class distribution
        #
        # H = -Σ p_i * log(p_i)
        #
        # Higher entropy = more uniform distribution
        # ------------------------------------------------------------------
        p = np.array([v / total for v in dist.values() if v > 0], dtype=float)
        entropy = float(-(p * np.log(p)).sum()) if p.size > 0 else 0.0

        # ------------------------------------------------------------------
        # Prepare summary
        # ------------------------------------------------------------------
        return {
            "total_annotations": total,
            "num_classes": len(categories),
            "num_classes_with_instances": len(nonzero),
            "num_classes_without_instances": len(categories) - len(nonzero),

            "distribution": dist,                # name → count
            "percent_distribution": percent_dist,

            "mean_per_class": mean_count,
            "std_per_class": std_count,

            "imbalance_ratio": float(imbalance),
            "entropy": entropy,
        }
