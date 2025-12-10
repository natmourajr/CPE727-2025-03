# src/dataset/validation/bbox_corrector.py
from __future__ import annotations
from typing import Dict, Any, List

class BBoxCorrector:
    """
    Correct COCO-style pixel absolute bounding boxes.
    
    COCO format: [xmin, ymin, width, height]
    """

    def correct(self, dataset: Dict[str, Any]) -> None:

        cleaned = []

        for ann in dataset["annotations"]:
            bbox = ann["bbox"]
            img_id = ann["image_id"]

            # obter dimensões da imagem
            img = next(i for i in dataset["images"] if i["id"] == img_id)
            W = img["width"]
            H = img["height"]

            fixed = self._fix_coco_bbox(bbox, W, H)

            if fixed is not None:
                ann["bbox"] = fixed
                cleaned.append(ann)

        dataset["annotations"] = cleaned


    def _fix_coco_bbox(self, bbox, img_w, img_h):
        xmin, ymin, w, h = bbox

        # Corrigir width e height negativos
        if w < 0:
            xmin = xmin + w
            w = abs(w)

        if h < 0:
            ymin = ymin + h
            h = abs(h)

        # garantir limites
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_w, xmin + w)
        ymax = min(img_h, ymin + h)

        # recalcular
        w = xmax - xmin
        h = ymax - ymin

        # descartar bbox degenerada
        if w < 1 or h < 1:
            return None

        return [xmin, ymin, w, h]
