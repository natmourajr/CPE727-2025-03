from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from PIL import Image


class CocoDatasetXYXY(CocoDetection):
    """
    COCO dataset rápido baseado no TorchVision.

    Suporta:
    ✔ TorchVision transforms (rápido)
    ✔ Albumentations transforms (opcional)
    """

    def __init__(self, img_dir, ann_file, transforms=None):
        super().__init__(img_dir, ann_file)
        self.transforms = transforms
        self.to_tensor = T.ToTensor()

        # Filtrar imagens sem bbox válida
        valid_ids = []
        removed = 0

        for img_id in self.ids:
            anns = self.coco.imgToAnns.get(img_id, [])
            valid = any(a["bbox"][2] > 1 and a["bbox"][3] > 1 for a in anns)
            if valid:
                valid_ids.append(img_id)
            else:
                removed += 1

        print(f"[CocoDatasetXYXY] Valid: {len(valid_ids)}, removed: {removed}")
        self.ids = valid_ids

    # ----------------------------------------------------------
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img_id = self.ids[idx]

        orig_w, orig_h = img.size

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transforms:
            transformed = self.transforms(
                image=np.array(img),
                bboxes=boxes,
                labels=labels
            )

            img = transformed["image"]

            # Albumentations -> np array → tensor
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)

        else:
            img = self.to_tensor(img).float()
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([orig_h, orig_w])
        }

        return img, target
