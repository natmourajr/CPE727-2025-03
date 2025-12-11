import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Any, Union, TypeAlias
from pydantic import Field, field_validator

from ..core.base_models import BaseModels, ModelsConfig
from ..core.enums import ModelTypeEnum

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]

# =========================
# CONFIG
# =========================
class LSTMConfig(ModelsConfig):
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.LSTM)
    input_size: int | None = Field(default=None)
    hidden_size: int = Field(..., gt=0)
    num_layers: int = Field(default=1, gt=0)
    output_size: int = Field(..., gt=0)

    @field_validator("input_size")
    @classmethod
    def check_input(cls, v):
        if v is not None and v <= 0:
            raise ValueError("input_size > 0 or None")
        return v

    def is_dynamic(self):
        return self.input_size is None

    def set_inferred(self, inp: int):
        object.__setattr__(self, "input_size", inp)


# =========================
# MODEL
# =========================
class LSTM(BaseModels, nn.Module):
    config: LSTMConfig

    def __init__(self, config: LSTMConfig):
        super().__init__(config=config)
        self.config = config
        self.lstm = None
        self.fc = None
        self._built = False

        if config.input_size is not None:
            self._build(config.input_size)

    def _build(self, input_size: int):
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.config.hidden_size, self.config.output_size)
        if self.config.is_dynamic():
            self.config.set_inferred(input_size)
        self._built = True

    # ======================
    # automatic reshape added
    # ======================
    def _ensure_input(self, x: torch.Tensor):
        # Build model on first batch
        if not self._built:
            self._build(x.shape[-1])

        # If 2D → (batch, seq_len=1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        return x

    def forward(self, x):
        x = self._ensure_input(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def fit(
        self, x_train, y_train=None, x_val=None, y_val=None,
        lr=1e-3, epochs=20, batch_size=32,
        patience=10, verbose=True
    ):
        """Supports both:
        - raw numpy/pandas arrays (x_train, y_train)
        - DataLoader objects (x_train is already a loader)
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # --------------------------------------------------------
        # CASE 1 — x_train IS ALREADY A DataLoader (used by Trainer)
        # --------------------------------------------------------
        if isinstance(x_train, DataLoader):
            train_loader = x_train
            val_loader = x_val

        else:
            # --------------------------------------------------------
            # CASE 2 — x_train is numpy/pandas → create loaders
            # --------------------------------------------------------
            if hasattr(x_train, "values"): x_train = x_train.values
            if hasattr(y_train, "values"): y_train = y_train.values
            if hasattr(x_val, "values"): x_val = x_val.values
            if hasattr(y_val, "values"): y_val = y_val.values

            train_ds = TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            val_ds = TensorDataset(
                torch.tensor(x_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        best_val = float("inf")
        bad_epochs = 0

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):

            # ------------------ TRAIN ------------------
            self.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            # ------------------ VALIDATION ------------------
            self.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss_total += criterion(self(xb), yb).item()

            val_loss = val_loss_total / len(val_loader)

            # save history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose:
                print(f"[LSTM] Epoch {epoch+1}/{epochs} "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # ------------------ EARLY STOPPING ------------------
            if (val_loss < 0.1) or (best_val<0.1):
                if val_loss < best_val:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        if verbose:
                            print(f"[LSTM] Early stopping at epoch {epoch+1}")
                        break

        history["best_val"] = best_val
        return history




    def predict(self, x, batch_size=1024):
        if hasattr(x, "values"):
            x = x.values

        device = next(self.parameters()).device
        self.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                x_batch = torch.tensor(x[i:i+batch_size], dtype=torch.float32).to(device)
                preds.append(torch.argmax(self(x_batch), dim=-1).cpu())
        return torch.cat(preds).numpy()
