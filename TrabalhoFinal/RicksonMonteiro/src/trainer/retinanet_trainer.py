from __future__ import annotations
from pathlib import Path
import json
import time
from typing import Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights

from torchvision.ops import batched_nms
from torch.cuda.amp import autocast, GradScaler

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.dataset.coco_dataset import CocoDatasetXYXY
from src.trainer.utils import collate_fn, save_metrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

from src.dataset.transforms.retina_transform import RetinaTransform

def warmup_lr(optimizer, step, warmup_steps, base_lr, start_factor=0.1):
    """Warmup linear por step: começa em base_lr * start_factor e sobe até base_lr."""
    if step >= warmup_steps:
        return

    alpha = step / warmup_steps
    scale = start_factor + alpha * (1.0 - start_factor)

    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


def get_retina_train_transforms(size=640):
    return T.Compose([
        T.Resize((size, size)),   # simples e consistente
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class RetinaNetTrainer:

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

    def create_model(self):
        num_classes = int(self.model_cfg.get("num_classes", )) +1
        pretrained = bool(self.model_cfg.get("pretrained", True))

        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None

        model = retinanet_resnet50_fpn_v2(weights=weights)

        # Replace classification head like torchvision recommends
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=model.backbone.out_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes
        )

        return model

    # -------------------------------------------------------------------------
    def get_dataloaders(self):
        img_dir = self.model_cfg.get("img_dir", "data/canonical/images/canonical")

        train_ds = CocoDatasetXYXY(img_dir, self.train_json)
        val_ds = CocoDatasetXYXY(img_dir, self.val_json)

        num_workers = int(self.training_cfg.get("num_workers", 0))

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=max(0, min(2, num_workers)),
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
    def postprocess(self, out):
        boxes = torch.tensor(out["boxes"], dtype=torch.float32)
        scores = torch.tensor(out["scores"], dtype=torch.float32)
        labels = torch.tensor(out["labels"], dtype=torch.int64)

        keep = scores >= self.score_threshold
        if keep.sum() == 0:
            return []

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        keep_idx = batched_nms(boxes, scores, labels, self.nms_iou)
        keep_idx = keep_idx[: self.max_dets]

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

        tmp_file = self.fold_dir / "tmp_retinanet_preds.json"
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

        # --------------------- OPTIMIZER ---------------------
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # --------------------- SCHEDULER ---------------------
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.scheduler_gamma
        )

        scaler = GradScaler()

        train_loader, val_loader = self.get_dataloaders()

        # --------------------- WARMUP CONFIG ---------------------
        warmup_epochs = 2
        warmup_steps = len(train_loader) * warmup_epochs
        global_step = 0
        base_lr = self.lr

        best_map = 0.0
        patience_count = 0

        history = {"train_loss": [], "map": [], "precision": [], "recall": [], "lr": []}

        print(f"\nIniciando treino RetinaNet — device: {self.device}")

        # --------------------- EPOCH LOOP ---------------------
        for epoch in range(1, self.num_epochs + 1):

            epoch_start = time.time()
            self.model.train()

            total_loss = 0.0
            iters = 0

            # --------------------- TRAIN LOOP ---------------------
            for imgs, tgts in train_loader:
                # imgs: list[PIL->tensor], tgts: list[dict]
                # primeiro mova imagens p/ device but keep pairing
                paired = []
                for img, tgt in zip(imgs, tgts):
                    # Some datasets might have boxes stored as list; ensure tensor
                    if "boxes" in tgt and len(tgt["boxes"]) > 0:
                        img_dev = img.to(self.device)
                        tgt_dev = {
                            "boxes": tgt["boxes"].to(self.device),
                            "labels": tgt["labels"].to(self.device)
                        }
                        paired.append((img_dev, tgt_dev))

                if len(paired) == 0:
                    continue

                imgs = [p[0] for p in paired]
                tgts = [p[1] for p in paired]

                self.optimizer.zero_grad()

                with autocast():
                    loss_dict = self.model(imgs, tgts)
                    loss = sum(loss_dict.values())

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # --------------------- WARMUP POR STEP ---------------------
                if global_step < warmup_steps:
                    warmup_lr(self.optimizer, global_step, warmup_steps, base_lr)

                global_step += 1

                total_loss += loss.item()
                iters += 1

            avg_loss = total_loss / iters if iters > 0 else float("nan")
            history["train_loss"].append(avg_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # --------------------- SCHEDULER APÓS WARMUP ---------------------
            if global_step >= warmup_steps:
                self.scheduler.step()

            # --------------------- VALIDATION ---------------------
            raw_preds = self._get_raw_predictions(self.model, val_loader)
            mAP, ap50, recall = self._compute_metrics_from_raw(raw_preds, self.val_json)

            history["map"].append(mAP)
            history["precision"].append(ap50)
            history["recall"].append(recall)

            print(f"Epoch {epoch}/{self.num_epochs} — Loss: {avg_loss:.4f} — mAP = {mAP:.4f}")
            save_metrics(self.fold_dir, history)

            torch.save(self.model.state_dict(), self.models_dir / "last.pth")

            if mAP > best_map:
                best_map = mAP
                patience_count = 0
                torch.save(self.model.state_dict(), self.models_dir / "best.pth")
                print("Novo melhor modelo salvo → best.pth")
            else:
                patience_count += 1
                print(f"Sem melhora ({patience_count}/{self.patience})")

            if patience_count >= self.patience:
                print("\nEarly stopping ativado.")
                break

        save_metrics(self.fold_dir, history)

        best_path = (
            self.models_dir / "best.pth"
            if (self.models_dir / "best.pth").exists()
            else (self.models_dir / "last.pth")
        )

        print(f"\nTreino finalizado! Melhor mAP={best_map:.4f}")
        return float(best_map), best_path
