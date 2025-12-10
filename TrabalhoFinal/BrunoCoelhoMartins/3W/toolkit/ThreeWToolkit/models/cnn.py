import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Any, Union, TypeAlias
from pydantic import Field, field_validator

from ..core.base_models import BaseModels, ModelsConfig
from ..core.enums import ModelTypeEnum, ActivationFunctionEnum

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]

# =========================
# CONFIG
# =========================
class CNNConfig(ModelsConfig):
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.CNN)
    input_channels: int | None = Field(default=None)
    conv_channels: tuple[int, ...] = Field(..., min_length=1)
    kernel_sizes: tuple[int, ...] = Field(..., min_length=1)
    output_size: int = Field(..., gt=0)
    activation_function: str = Field(default="relu")

    @field_validator("activation_function")
    @classmethod
    def check_activation(cls, v):
        valid = {"relu", "sigmoid", "tanh"}
        if v.lower() not in valid:
            raise ValueError(f"activation_function must be one of {valid}")
        return v.lower()

    def is_dynamic(self):
        return self.input_channels is None

    def set_inferred(self, c: int):
        object.__setattr__(self, "input_channels", c)


# =========================
# MODEL
# =========================
class CNN(BaseModels, nn.Module):
    config: CNNConfig

    def __init__(self, config: CNNConfig):
        super().__init__(config=config)
        self.config = config
        self.activation = self._get_activation(config.activation_function)
        self.conv = None
        self.fc = None
        self._built = False

    def _get_activation(self, act: str):
        act = act.lower()
        if act == "relu":
            return nn.ReLU()
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {act}")

    def _build(self, in_channels: int, seq_len: int):
        layers = []
        c = in_channels
        for out_c, k in zip(self.config.conv_channels, self.config.kernel_sizes):
            layers.append(nn.Conv1d(c, out_c, kernel_size=k, padding=k//2))
            layers.append(self.activation)
            c = out_c
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            flat_dim = self.conv(dummy).numel()

        self.fc = nn.Linear(flat_dim, self.config.output_size)

        if self.config.is_dynamic():
            self.config.set_inferred(in_channels)

        self._built = True

    def _ensure(self, x):
        if not self._built:
            if x.ndim == 2:
                x = x.unsqueeze(1)
            self._build(x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        x = self._ensure(x)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.conv(x)
        return self.fc(feat.flatten(1))

    # =========================
    # FIXED FIT FUNCTION
    # =========================
    def fit(
        self, x_train, y_train=None, x_val=None, y_val=None,
        lr=1e-3, epochs=20, batch_size=32,
        patience=10, verbose=True
    ):
        """
        CNN fit that supports both:
        - x_train as DataLoader (used by Trainer)
        - x_train as numpy/pandas arrays
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # --------------------------------------------------------
        # CASE 1 – Train/Val already come as DataLoaders (Trainer)
        # --------------------------------------------------------
        if isinstance(x_train, DataLoader):

            train_loader = x_train

            # ❗ FIX: val_loader must be the loader, NOT the raw x_val array
            if isinstance(x_val, DataLoader):
                val_loader = x_val
            else:
                val_loader = None

        else:
            # --------------------------------------------------------
            # CASE 2 – Raw arrays → convert
            # --------------------------------------------------------
            if hasattr(x_train, "values"): x_train = x_train.values
            if hasattr(y_train, "values"): y_train = y_train.values
            if hasattr(x_val, "values"): x_val = x_val.values
            if hasattr(y_val, "values"): y_val = y_val.values

            train_ds = TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            if x_val is not None and y_val is not None:
                val_ds = TensorDataset(
                    torch.tensor(x_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long)
                )
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            else:
                val_loader = None

        # --------------------------------------------------------
        # FORCE MODEL BUILD BEFORE OPTIMIZER
        # --------------------------------------------------------
        xb, _ = next(iter(train_loader))
        xb = xb.to(device)
        _ = self(xb)  # triggers _ensure() → _build()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        best_val = float("inf")
        bad_epochs = 0

        history = {"train_loss": [], "val_loss": []}

        # --------------------------------------------------------
        # TRAINING LOOP
        # --------------------------------------------------------
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
            history["train_loss"].append(train_loss)

            # ------------------ VALIDATION ------------------
            if val_loader is not None:
                self.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_loss_total += criterion(self(xb), yb).item()

                val_loss = val_loss_total / len(val_loader)
                history["val_loss"].append(val_loss)

                if verbose:
                    print(f"[CNN] Epoch {epoch+1}/{epochs} "
                        f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

                # Early stopping
                if val_loss < best_val:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        if verbose:
                            print(f"[CNN] Early stopping at epoch {epoch+1}")
                        break
            else:
                # No validation
                history["val_loss"].append(None)
                if verbose:
                    print(f"[CNN] Epoch {epoch+1}/{epochs} Train: {train_loss:.4f}")

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
