from __future__ import annotations
import json
import time
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainerBase(ABC):
    """
    Base Trainer com utilitários comuns.
    Subclasses devem implementar:
      - create_model()
      - get_dataloaders(train_json, val_json)
      - postprocess(raw_pred, orig_size) -> list[coco_det]
    """

    def __init__(self,
                 training_cfg: Dict[str, Any],
                 model_cfg: Dict[str, Any],
                 fold_dir: Path,
                 train_json: str,
                 val_json: str):
        self.training_cfg = training_cfg or {}
        self.model_cfg = model_cfg or {}
        self.fold_dir = Path(fold_dir)
        self.fold_dir.mkdir(parents=True, exist_ok=True)

        self.train_json = train_json
        self.val_json = val_json

        # model / training hyperparams
        self.batch_size = int(self.training_cfg.get("batch_size", 8))
        self.num_epochs = int(self.training_cfg.get("num_epochs",  200))
        self.lr = float(self.training_cfg.get("learning_rate", 0.001))
        self.weight_decay = float(self.training_cfg.get("weight_decay", 5e-4))
        self.momentum = float(self.training_cfg.get("momentum", 0.9))
        self.warmup_epochs = int(self.training_cfg.get("warmup_epochs", 0))
        self.scheduler_step = int(self.training_cfg.get("scheduler_step_size", 10))
        self.scheduler_gamma = float(self.training_cfg.get("scheduler_gamma", 0.1))
        self.patience = int(self.training_cfg.get("early_stop", {}).get("patience", 15))

        # paths
        self.models_dir = self.fold_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # state
        self.device = DEVICE
        self.model = None
        self.optimizer = None
        self.scheduler = None

    @abstractmethod
    def create_model(self):
        """Return a torch.nn.Module ready (unmoved to device)."""
        raise NotImplementedError

    @abstractmethod
    def get_dataloaders(self):
        """Return (train_loader, val_loader)."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, raw_pred: Dict[str, Any], orig_size):
        """Convert raw model pred -> list of COCO-format detections for evaluation."""
        raise NotImplementedError

    # --------------------
    def apply_warmup(self, epoch):
        if self.warmup_epochs <= 0:
            return
        if epoch <= self.warmup_epochs:
            warm_lr = self.lr * (epoch / max(1, self.warmup_epochs))
            for g in self.optimizer.param_groups:
                g["lr"] = warm_lr

    def save_checkpoint(self, name="last.pth"):
        path = self.models_dir / name
        torch.save(self.model.state_dict(), path)

    def save_best(self, name="best_map.pth"):
        path = self.models_dir / name
        torch.save(self.model.state_dict(), path)

    def run(self):
        """Main training loop. Returns metrics dict (history)."""
        # build model, optimizer, scheduler
        self.model = self.create_model().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                   momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)

        train_loader, val_loader = self.get_dataloaders()

        best_map = 0.0
        patience_count = 0

        history = {"train_loss": [], "map": [], "lr": []}

        print(f"\n🚀 Starting training ({self.model_cfg.get('name', 'model')}) on device {self.device}\n")

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            self.model.train()

            self.apply_warmup(epoch)

            total_loss = 0.0
            iters = 0

            for imgs, tgts in train_loader:
                imgs = [img.to(self.device) for img in imgs]
                targets_list = [{"boxes": t["boxes"].to(self.device), "labels": t["labels"].to(self.device)} for t in tgts if len(t["boxes"]) > 0]
                if len(targets_list) == 0:
                    continue

                self.optimizer.zero_grad()
                loss_dict = self.model(imgs, targets_list)
                loss = sum(loss_dict.values())
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                iters += 1

            avg_loss = total_loss / iters if iters > 0 else float("nan")
            history["train_loss"].append(avg_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{self.num_epochs} — Loss: {avg_loss:.4f} — time: {epoch_time:.1f}s")

            self.scheduler.step()

            # validation -> raw preds + COCOeval
            raw_preds = self._get_raw_predictions(val_loader)
            current_map = self._compute_map_from_raw(raw_preds)
            history["map"].append(current_map)
            print(f"✔ Epoch {epoch} — mAP = {current_map:.4f}")

            # checkpointing
            self.save_checkpoint("last.pth")
            if current_map > best_map:
                best_map = current_map
                patience_count = 0
                self.save_best("best_map.pth")
                print("🏆 Novo melhor modelo salvo (por mAP)")
            else:
                patience_count += 1
                print(f"Sem melhora ({patience_count}/{self.patience})")

            if patience_count >= self.patience:
                print("\n⛔ Early stopping ativado.")
                break

        # save metrics
        json.dump(history, open(self.fold_dir / "metrics.json", "w"), indent=2)
        print("\n🏁 Treinamento finalizado!")
        print(f"Melhor mAP: {best_map:.4f}")
        return {"best_map": best_map, "history": history}

    # --------------------
    @torch.no_grad()
    def _get_raw_predictions(self, loader):
        """Run inference on val set and collect raw preds."""
        self.model.eval()
        outputs = []
        for imgs, tgts in loader:
            imgs = [img.to(self.device) for img in imgs]
            preds = self.model(imgs)

            for pred, tgt in zip(preds, tgts):
                outputs.append({
                    "image_id": int(tgt["image_id"].item()),
                    "orig_size": tgt["orig_size"].cpu().tolist(),
                    "boxes": pred["boxes"].cpu().tolist(),
                    "scores": pred["scores"].cpu().tolist(),
                    "labels": pred["labels"].cpu().tolist()
                })
        return outputs

    def _compute_map_from_raw(self, raw_preds):
        """
        Default implementation relies on self.postprocess for each image.
        Returns AP@[.50:.95].
        """
        # load GT
        from pycocotools.coco import COCO
        coco_gt = COCO(self.val_json)

        results = []
        for pred in raw_preds:
            dets = self.postprocess(pred, score_threshold=self.training_cfg.get("score_threshold", 0.001),
                                    nms_iou=self.training_cfg.get("nms_iou", 0.5),
                                    max_dets=self.training_cfg.get("max_dets", 200))
            for d in dets:
                d["image_id"] = pred["image_id"]
                results.append(d)

        if len(results) == 0:
            return 0.0

        tmp_file = self.fold_dir / "tmp_preds.json"
        json.dump(results, open(tmp_file, "w"), indent=2)

        coco_dt = coco_gt.loadRes(str(tmp_file))
        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return float(coco_eval.stats[0])
