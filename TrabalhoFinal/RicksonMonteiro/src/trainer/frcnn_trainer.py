from __future__ import annotations
from pathlib import Path
import json
import time
from typing import Tuple, Dict, Any, List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import batched_nms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler

from src.dataset.coco_dataset import CocoDatasetXYXY
from src.trainer.utils import collate_fn, save_metrics



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

class FasterRCNNTrainer:
    """
    Trainer específico para Faster R-CNN.
        - Ao final de `train()` retorna (best_map: float, best_model_path: Path)
        - Salva metrics.json em fold_dir
        - Salva checkpoints em fold_dir/models/{last.pth, best.pth}
    """

    def __init__(
        self,
        training_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
        fold_dir: Path,
        train_json: str,
        val_json: str,
    ):
        self.training_cfg = training_cfg or {}
        self.model_cfg = model_cfg or {}
        self.fold_dir = Path(fold_dir)
        self.fold_dir.mkdir(parents=True, exist_ok=True)

        # paths to ann files
        self.train_json = str(train_json)
        self.val_json = str(val_json)

        # hyperparams (order of precedence: model_cfg -> training_cfg -> default)
        self.batch_size = int(self.training_cfg.get("batch_size", 16))
        self.num_epochs = int(self.training_cfg.get("num_epochs", 100))
        self.lr = float(self.training_cfg.get("learning_rate", 0.01))
        self.weight_decay = float(self.training_cfg.get("weight_decay", 5e-4))
        self.momentum = float(self.training_cfg.get("momentum", 0.9))
        self.warmup_epochs = int(self.training_cfg.get("warmup_epochs", 0))
        self.scheduler_step = int(self.training_cfg.get("scheduler_step_size", 10))
        self.scheduler_gamma = float(self.training_cfg.get("scheduler_gamma", 0.97))
        self.patience = int(self.training_cfg.get("early_stop", {}).get("patience", 10))

        self.score_threshold = float(self.training_cfg.get("score_threshold", 0.05))
        self.nms_iou = float(self.training_cfg.get("nms_iou", 0.5))
        self.max_dets = int(self.training_cfg.get("max_dets", 200))

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model artifacts dir (per fold)
        self.models_dir = self.fold_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # internal state
        self.model = None
        self.optimizer = None
        self.scheduler = None

    # ----------------------------
    def create_model(self) -> torch.nn.Module:
        """Cria fasterrcnn_resnet50_fpn e substitui predictor por num_classes."""

        num_classes = int(self.model_cfg.get("num_classes", )) +1 # including background
        pretrained = bool(self.model_cfg.get("pretrained", True))

        # load backbone weights if pretrained requested;
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)

        # replace box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    # ----------------------------
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Usa CocoDatasetXYXY com image_dir vindo do model_cfg ou default 'data/canonical/images/canonical'.
        Espera que train_json/val_json estejam completos (COCO-style).
        """
        img_dir = self.model_cfg.get("img_dir", "data/canonical/images/canonical")
        train_ds = CocoDatasetXYXY(img_dir, self.train_json, transforms=None)
        val_ds = CocoDatasetXYXY(img_dir, self.val_json, transforms=None)

        num_workers = int(self.training_cfg.get("num_workers", 0))

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=max(0, min(2, num_workers)),
            pin_memory=True
        )

        return train_loader, val_loader
    
    @torch.no_grad()
    def _get_raw_predictions(self, model: torch.nn.Module, loader: DataLoader) -> List[Dict[str, Any]]:
        """Inference loop: retorna lista de dicts {image_id, orig_size, boxes, scores, labels}."""
        model.eval()
        outputs = []
        for imgs, tgts in loader:
            imgs = [img.to(self.device) for img in imgs]
            preds = model(imgs)
            for pred, tgt in zip(preds, tgts):
                outputs.append({
                    "image_id": int(tgt["image_id"].item()),
                    "orig_size": tgt["orig_size"].cpu().tolist(),  # [h, w]
                    "boxes": pred["boxes"].cpu().tolist(),
                    "scores": pred["scores"].cpu().tolist(),
                    "labels": pred["labels"].cpu().tolist()
                })
        return outputs

    # ----------------------------
    def postprocess(self, out: Dict[str, Any], score_threshold=None, nms_iou=None, max_dets=None) -> List[Dict[str, Any]]:
        """Postprocess padrão para Faster R-CNN (boxes já no espaço original)."""
        score_threshold = self.score_threshold if score_threshold is None else score_threshold
        nms_iou = self.nms_iou if nms_iou is None else nms_iou
        max_dets = self.max_dets if max_dets is None else max_dets

        boxes = torch.tensor(out["boxes"], dtype=torch.float32)
        scores = torch.tensor(out["scores"], dtype=torch.float32)
        labels = torch.tensor(out["labels"], dtype=torch.int64)

        # filtro por score
        keep = scores >= score_threshold
        if keep.sum() == 0:
            return []

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        keep_idx = batched_nms(boxes, scores, labels, nms_iou)
        keep_idx = keep_idx[:max_dets]

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        results = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            results.append({
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
                "category_id": int(label)
            })
        return results

    # ----------------------------
    def _compute_metrics_from_raw(self, raw_preds: List[Dict[str, Any]], val_json_path: str) -> Dict[str, float]:
        """
        Converte preds para COCO, executa COCOeval e retorna um dict com:
            - mAP:   AP@[.50:.95]
            - ap50:  AP@0.50
            - recall: AR@100
        """

        results = []
        for pred in raw_preds:
            dets = self.postprocess(pred)
            for d in dets:
                d["image_id"] = pred["image_id"]
                results.append(d)

        # Nenhuma detecção → métricas zero
        if len(results) == 0:
            return {"mAP": 0.0, "AP50": 0.0, "AR": 0.0}

        # Arquivo temporário para COCOeval
        tmp_file = self.fold_dir / "tmp_frcnn_preds.json"
        json.dump(results, open(tmp_file, "w"), indent=2)

        # Rodar COCOeval
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

    # ----------------------------
    def save_checkpoint(self, name="last.pth"):
        path = self.models_dir / name
        torch.save(self.model.state_dict(), path)
        return path

        # ----------------------------
    def train(self) -> Tuple[float, Path]:
        self.model = self.create_model().to(self.device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.97
        )

        scaler = GradScaler()

        train_loader, val_loader = self.get_dataloaders()

        # ------------------- WARMUP CONFIG -------------------
        warmup_epochs = 2
        warmup_steps = len(train_loader) * warmup_epochs
        global_step = 0
        base_lr = self.lr

        best_map = 0.0
        patience_count = 0

        history = {"train_loss": [], "map": [], "precision": [], "recall": [], "lr": []}

        print(f"\nIniciando treino FRCNN — device: {self.device}")

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            self.model.train()

            total_loss = 0.0
            iters = 0

            # ---------------------- TRAIN LOOP ----------------------
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

                # ---------------------- WARMUP POR STEP ----------------------
                if global_step < warmup_steps:
                    warmup_lr(self.optimizer, global_step, warmup_steps, base_lr)

                global_step += 1

                total_loss += loss.item()
                iters += 1

            avg_loss = total_loss / iters if iters > 0 else float("nan")
            history["train_loss"].append(avg_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])
            epoch_time = time.time() - epoch_start

            # scheduler só roda DEPOIS do warmup terminar
            if global_step >= warmup_steps:
                self.scheduler.step()

            # ---------------------- VALIDATION ----------------------
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

        save_metrics(self.fold_dir, history)

        best_path = (
            self.models_dir / "best.pth"
            if (self.models_dir / "best.pth").exists()
            else (self.models_dir / "last.pth")
        )

        print(f"\nTreino finalizado! Melhor mAP={best_map:.4f}")
        return float(best_map), best_path
