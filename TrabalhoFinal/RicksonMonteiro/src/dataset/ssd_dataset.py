from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO


class SSDCocoDatasetFinal(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

        valid, removed = [], 0
        for img_id in self.ids:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            if any(a["bbox"][2] > 1 and a["bbox"][3] > 1 for a in anns):
                valid.append(img_id)
            else:
                removed += 1

        self.ids = valid
        print(f"[SSD Dataset] Valid={len(valid)} Removed={removed}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        info = self.coco.loadImgs(img_id)[0]
        file_name = info["file_name"]
        img_path = str(self.img_dir / file_name)

        img_raw = cv2.imread(img_path)
        if img_raw is None:
            raise FileNotFoundError(img_path)

        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_raw.shape[:2]

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x+w, y+h])
            labels.append(int(ann["category_id"]))

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        if self.transforms:
            transformed = self.transforms(
                image=img_raw,
                bboxes=boxes,
                class_labels=labels
            )
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        else:
            from albumentations.pytorch import ToTensorV2
            img = ToTensorV2()(image=img_raw)["image"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([orig_h, orig_w])
        }

        return img, target
