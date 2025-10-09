# mlp_mnist.py
# -*- coding: utf-8 -*-
"""
Implementa:
 - classe `data_loader` para baixar/carregar o MNIST com DataLoaders de treino/val/teste
 - classe `MLPClassifier` (PyTorch) com fit, predict, save_model e load_model
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------------------
# Utilidades
# ---------------------------

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # para reprodutibilidade melhor (custo: pode reduzir performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


# ---------------------------
# Data Loader: baixa e provê DataLoaders
# ---------------------------

@dataclass
class DataConfig:
    data_dir: str = "../Data"
    batch_size: int = 128
    val_split: float = 0.1
    num_workers: int = 2
    seed: int = 42
    normalize: bool = True


class data_loader:
    """
    Classe responsável por baixar e carregar o banco MNIST e expor DataLoaders.
    Exemplo:
        dl = data_loader(DataConfig(data_dir="./data"))
        train_loader, val_loader, test_loader = dl.get_loaders()
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        set_seed(self.config.seed)

        # Transformações: tensor + (opcional) normalização padrão do MNIST
        tfms = [transforms.ToTensor()]
        if self.config.normalize:
            tfms.append(transforms.Normalize((0.1307,), (0.3081,)))
        self.transform = transforms.Compose(tfms)

        # Baixa/Carrega datasets
        self.train_dataset = datasets.MNIST(
            root=self.config.data_dir, train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            root=self.config.data_dir, train=False, download=True, transform=self.transform
        )

        # Split treino/val
        val_len = int(len(self.train_dataset) * self.config.val_split)
        train_len = len(self.train_dataset) - val_len
        self.train_subset, self.val_subset = random_split(
            self.train_dataset,
            lengths=[train_len, val_len],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def get_loaders(self) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        train_loader = DataLoader(
            self.train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        ) if len(self.val_subset) > 0 else None
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader


# ---------------------------
# Modelo MLP para MNIST
# ---------------------------

class MLPClassifier(nn.Module):
    """
    Rede MLP simples para MNIST (28x28 -> 784 -> ... -> 10).
    Métodos:
      - fit(train_loader, val_loader, epochs, ...)
      - predict(dataloader, return_probs=True)
      - save_model(path)
      - load_model(path)  [classmethod -> retorna instância carregada]
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [256, 128],
        output_size: int = 10,
        dropout: float = 0.2,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        set_seed(seed)

        layers: List[nn.Module] = []
        last = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)])
            last = h
        layers.append(nn.Linear(last, output_size))
        self.net = nn.Sequential(*layers)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.seed = seed

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28) -> flatten em (B, 784)
        if x.ndim == 4:
            x = torch.flatten(x, start_dim=1)
        return self.net(x)

    # ---------------------------
    # Treinamento
    # ---------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        log_every: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Treina o modelo. Retorna histórico com loss/acc de treino e (se houver) validação.
        """
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        global_step = 0
        for epoch in range(1, epochs + 1):
            running_loss, running_acc, seen = 0.0, 0.0, 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self(images)
                loss = criterion(logits, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                running_acc += accuracy_from_logits(logits, targets) * batch_size
                seen += batch_size
                global_step += 1

                if log_every and (batch_idx + 1) % log_every == 0:
                    print(
                        f"[Epoch {epoch}/{epochs}] Step {batch_idx+1}/{len(train_loader)} "
                        f"Loss: {loss.item():.4f}"
                    )

            epoch_train_loss = running_loss / max(seen, 1)
            epoch_train_acc = running_acc / max(seen, 1)
            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)

            # Validação
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                print(
                    f"Epoch {epoch}: train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={epoch_train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch}: train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={epoch_train_acc:.4f}"
                )

        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.eval()
        total_loss, total_acc, total = 0.0, 0.0, 0
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self(images)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(logits, targets) * batch_size
            total += batch_size
        self.train()
        return total_loss / max(total, 1), total_acc / max(total, 1)

    # ---------------------------
    # Predição
    # ---------------------------
    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        return_probs: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Passa o dado pelo modelo e retorna:
            - y_pred (labels inteiros)
            - y_prob (probabilidades por classe) se return_probs=True
        """
        self.eval()
        all_preds = []
        all_probs = [] if return_probs else None

        for images, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            logits = self(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu())

            if return_probs:
                probs = torch.softmax(logits, dim=1).detach().cpu()
                all_probs.append(probs)

        y_pred = torch.cat(all_preds, dim=0)
        y_prob = torch.cat(all_probs, dim=0) if return_probs else None
        self.train()
        return y_pred, y_prob

    # ---------------------------
    # Checkpoint (salvar/carregar)
    # ---------------------------
    def save_model(self, path: str) -> None:
        """
        Salva pesos e hiperparâmetros mínimos para reconstruir o modelo.
        Cria:
          - path (arquivo .pt)  -> state_dict + meta compactados
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "meta": {
                "input_size": self.input_size,
                "hidden_sizes": self.hidden_sizes,
                "output_size": self.output_size,
                "dropout": self.dropout,
                "seed": self.seed,
            },
        }
        torch.save(checkpoint, path)
        print(f"Modelo salvo em: {path}")

    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None) -> "MLPClassifier":
        """
        Carrega o modelo a partir de um arquivo salvo por save_model e retorna a instância.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {path}")

        checkpoint = torch.load(path, map_location=device or "cpu")
        meta = checkpoint["meta"]
        model = cls(
            input_size=meta["input_size"],
            hidden_sizes=meta["hidden_sizes"],
            output_size=meta["output_size"],
            dropout=meta["dropout"],
            seed=meta.get("seed", 42),
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Modelo carregado de: {path}")
        return model
