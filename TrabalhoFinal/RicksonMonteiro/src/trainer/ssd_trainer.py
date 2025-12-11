from __future__ import annotations
from pathlib import Path
import json
import time
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models import MobileNet_V3_Large_Weights

from torchvision.ops import batched_nms
from torch.cuda.amp import autocast, GradScaler

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset.ssd_dataset import SSDCocoDatasetFinal
from src.trainer.utils import collate_fn, save_metrics


def get_train_transforms(size=320):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_area=1,
        min_visibility=0.2
    ))


def get_val_transforms(size=320):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

def warmup_lr(optimizer, step, warmup_steps, base_lr, start_factor=0.1):
    """
    Ajusta o LR durante o warmup.
    - Começa com base_lr * start_factor e sobe linearmente até base_lr.
    """
    if step >= warmup_steps:
        return

    alpha = step / warmup_steps
    scale = start_factor + alpha * (1.0 - start_factor)

    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


class SSDLiteTrainer:
    """
    Trainer NO MESMO PADRÃO do RetinaNetTrainer,
    mas com transforms Albumentations + letterbox do script anterior.
    """

    def __init__(self, training_cfg, model_cfg, fold_dir, train_json, val_json):
        self.training_cfg = training_cfg or {}
        self.model_cfg = model_cfg or {}

        self.fold_dir = Path(fold_dir)
        self.fold_dir.mkdir(parents=True, exist_ok=True)

        self.train_json = str(train_json)
        self.val_json = str(val_json)

        # ---------------- HYPERPARAMS ----------------
        self.batch_size = int(self.training_cfg.get("batch_size", 16))
        self.num_epochs = int(self.training_cfg.get("num_epochs", 100))
        self.lr = float(self.training_cfg.get("learning_rate", 0.01))

        self.weight_decay = float(self.training_cfg.get("weight_decay", 5e-4))
        self.momentum = float(self.training_cfg.get("momentum", 0.9))

        self.scheduler_step = int(self.training_cfg.get("scheduler_step_size", 10))
        self.scheduler_gamma = float(self.training_cfg.get("scheduler_gamma", 0.97))
        self.warmup_epochs = int(self.training_cfg.get("warmup_epochs", 3))
        self.patience = int(self.training_cfg.get("early_stop", {}).get("patience", 10))

        self.score_threshold = float(self.training_cfg.get("score_threshold", 0.05))
        self.nms_iou = float(self.training_cfg.get("nms_iou", 0.5))
        self.max_dets = int(self.training_cfg.get("max_dets", 200))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models_dir = self.fold_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.img_size = int(self.model_cfg.get("img_size", 320))

    # -------------------------------------------------------------------------
    def _replace_ssd_head(self, model, num_classes):
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
            num_classes=num_classes
        )
        return model

    # -------------------------------------------------------------------------
    def create_model(self):
        num_classes = int(self.model_cfg.get("num_classes", )) + 1

        backbone_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2

        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes
        )

        # model = self._replace_ssd_head(model, num_classes)
        return model

    # -------------------------------------------------------------------------
    def get_dataloaders(self):
        img_dir = self.model_cfg.get("img_dir", "data/images")

        train_ds = SSDCocoDatasetFinal(
            img_dir,
            self.train_json,
            transforms=get_train_transforms(self.img_size)
        )
        val_ds = SSDCocoDatasetFinal(
            img_dir,
            self.val_json,
            transforms=get_val_transforms(self.img_size)
        )

        num_workers = int(self.training_cfg.get("num_workers", 0))

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=max(0, min(2, num_workers)),
            pin_memory=True,
        )

        return train_loader, val_loader

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _get_raw_predictions(self, model, loader):
        model.eval()
        outputs = []

        for imgs, tgts in loader:
            imgs = [img.to(self.device) for img in imgs]
            preds = model(imgs)

            for pred, tgt in zip(preds, tgts):
                outputs.append({
                    "image_id": int(tgt["image_id"]),
                    "orig_size": tgt["orig_size"].tolist(),
                    "boxes": pred["boxes"].cpu().tolist(),
                    "scores": pred["scores"].cpu().tolist(),
                    "labels": pred["labels"].cpu().tolist(),
                })
        return outputs

    # -------------------------------------------------------------------------
    def _restore_boxes_letterbox(self, boxes, orig_size):
        """ desfaz o letterbox exatamente como no script anterior """
        orig_h, orig_w = orig_size

        scale = min(self.img_size / orig_w, self.img_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= scale
        return boxes

    # -------------------------------------------------------------------------
    def postprocess(self, out):
        boxes = torch.tensor(out["boxes"], dtype=torch.float32)
        scores = torch.tensor(out["scores"], dtype=torch.float32)
        labels = torch.tensor(out["labels"], dtype=torch.int64)

        if boxes.numel() == 0:
            return []

        # remove padding do letterbox
        boxes = self._restore_boxes_letterbox(boxes, out["orig_size"])

        keep = scores >= self.score_threshold
        if keep.sum() == 0:
            return []

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        keep_idx = batched_nms(boxes, scores, labels, self.nms_iou)
        keep_idx = keep_idx[:self.max_dets]

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        results = []
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b.tolist()
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                results.append({
                    "bbox": [x1, y1, w, h],
                    "score": float(s),
                    "category_id": int(l),
                })
        return results

    # -------------------------------------------------------------------------
    def _compute_metrics_from_raw(self, raw_preds, val_json_path):
        results = []
        for pred in raw_preds:
            dets = self.postprocess(pred)
            for d in dets:
                d["image_id"] = pred["image_id"]
                results.append(d)

        if len(results) == 0:
            return 0.0, 0.0, 0.0

        tmp_file = self.fold_dir / "tmp_ssd_preds.json"
        json.dump(results, open(tmp_file, "w"), indent=2)

        coco_gt = COCO(val_json_path)
        coco_dt = coco_gt.loadRes(str(tmp_file))

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP = float(coco_eval.stats[0])
        ap50 = float(coco_eval.stats[1])
        recall = float(coco_eval.stats[8])

        return mAP, ap50, recall

    def train(self) -> Tuple[float, Path]:
        self.model = self.create_model().to(self.device)

        # ---------------- OPTIMIZER ----------------
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # ---------------- SCHEDULER NOVO (EXPONENTIAL) ----------------
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.97
        )

        scaler = GradScaler()

        train_loader, val_loader = self.get_dataloaders()


        warmup_steps = len(train_loader) * self.warmup_epochs
        global_step = 0
        base_lr = self.lr

        # ---------- LOGS ----------
        history = {"train_loss": [], "map": [], "precision": [], "recall": [], "lr": []}

        best_map = 0.0
        patience_count = 0

        print(f"\nIniciando treino SSDLite — device: {self.device}")

        # -------------------------------------------------------------------------
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            self.model.train()

            total_loss = 0.0
            iters = 0

            # ===================== TRAIN LOOP ======================
            for imgs, tgts in train_loader:
                imgs = [img.to(self.device) for img in imgs]

                tgts = [
                    {"boxes": t["boxes"].to(self.device), "labels": t["labels"].to(self.device)}
                    for t in tgts if len(t["boxes"]) > 0
                ]
                if len(tgts) == 0:
                    continue

                self.optimizer.zero_grad()

                with autocast():
                    loss_dict = self.model(imgs, tgts)
                    loss = sum(loss_dict.values())

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # ---------------------- WARMUP ----------------------
                if global_step < warmup_steps:
                    warmup_lr(self.optimizer, global_step, warmup_steps, base_lr)

                global_step += 1

                total_loss += loss.item()
                iters += 1

            avg_loss = total_loss / iters if iters > 0 else float("nan")
            history["train_loss"].append(avg_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # ------------------- SCHEDULER DEPOIS DO WARMUP -------------------
            if global_step >= warmup_steps:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # ===================== VALIDATION ======================
            raw_preds = self._get_raw_predictions(self.model, val_loader)
            current_map, precision, recall = self._compute_metrics_from_raw(raw_preds, self.val_json)

            history["map"].append(current_map)
            history["precision"].append(precision)
            history["recall"].append(recall)

            print(f"Epoch {epoch}/{self.num_epochs} — Loss: {avg_loss:.4f} — mAP = {current_map:.4f} — time: {epoch_time:.1f}s")
            save_metrics(self.fold_dir, history)

            torch.save(self.model.state_dict(), self.models_dir / "last.pth")

            if current_map > best_map:
                best_map = current_map
                patience_count = 0
                torch.save(self.model.state_dict(), self.models_dir / "best.pth")
                print("Novo melhor modelo salvo → best.pth")
            else:
                patience_count += 1
                print(f"Sem melhora ({patience_count}/{self.patience})")

            if patience_count >= self.patience:
                print("\nEarly stopping ativado.")
                break

        # ------------------- FINAL -----------------------------
        save_metrics(self.fold_dir, history)

        best_path = (
            self.models_dir / "best.pth"
            if (self.models_dir / "best.pth").exists()
            else (self.models_dir / "last.pth")
        )

        print(f"\nTreino finalizado! Melhor mAP={best_map:.4f}")
        return float(best_map), best_path
