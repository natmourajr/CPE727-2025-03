from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict


class CooccurrenceMatrix:
    """
    Compute class co-occurrence matrix.

    Responsibilities:
        - Track which classes appear together in each image
        - Produce a matrix with counts per pair (i, j)
        - Convert matrix to a human-readable format (using class names)

    Definition:
        matrix[A][B] = number of images where class A and class B appear together.

    Behavior:
        - Diagonal values represent number of images where the class appears.
        - Zero if classes never co-occur.
    """

    def compute(
        self,
        images: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        # -----------------------------------------------------------
        # No annotations → empty co-occurrence matrix
        # -----------------------------------------------------------
        if not annotations or not categories:
            return {}

        # -----------------------------------------------------------
        # Build mapping: image_id → set(category_ids)
        # -----------------------------------------------------------
        image_to_classes = defaultdict(set)
        for ann in annotations:
            image_to_classes[ann["image_id"]].add(ann["category_id"])

        # -----------------------------------------------------------
        # Prepare matrix with all class_ids initialized
        # -----------------------------------------------------------
        class_ids = [c["id"] for c in categories]
        matrix = {
            cid: {cid2: 0 for cid2 in class_ids}
            for cid in class_ids
        }

        # -----------------------------------------------------------
        # Populate the co-occurrence counts
        # -----------------------------------------------------------
        for class_set in image_to_classes.values():
            class_list = list(class_set)
            for i in range(len(class_list)):
                for j in range(len(class_list)):
                    ci = class_list[i]
                    cj = class_list[j]
                    matrix[ci][cj] += 1

        # -----------------------------------------------------------
        # Convert IDs → names for readability
        # -----------------------------------------------------------
        id_to_name = {c["id"]: c["name"] for c in categories}

        readable = {
            id_to_name[cid]: {
                id_to_name[cid2]: matrix[cid][cid2]
                for cid2 in class_ids
            }
            for cid in class_ids
        }

        return readable
