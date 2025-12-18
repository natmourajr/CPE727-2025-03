# run_models_memmap_with_results_with_test.py
# Stream + window the 3W dataset into memmaps and train LSTM, GRU, CNN safely.
# Splitting occurs at the event level BEFORE windowing to avoid leakage.
# Includes test evaluation and progress bars during windowing.

import os
import time
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch.nn as nn

from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.preprocessing import Windowing
from ThreeWToolkit.core.base_preprocessing import WindowingConfig
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig
from ThreeWToolkit.models import LSTMConfig, LSTM, GRUConfig, GRU, CNNConfig, CNN, MLPConfig, MLP
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.assessment.model_assess import ModelAssessment
from ThreeWToolkit.core.enums import TaskType, EventPrefixEnum

# -----------------------------
# USER SETTINGS
# -----------------------------
DATASET_PATH = Path("../../dataset")
WINDOW_SIZE = 100
WINDOW_OVERLAP = 0.5
PAD_LAST = True

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TMP_DIR = Path("./tmp_windows")
TMP_DIR.mkdir(exist_ok=True)

RESULTS_DIR = TMP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RUN_LSTM = True
RUN_GRU = True
RUN_CNN = True
RUN_MLP = True

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
# -----------------------------

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------- Load dataset ----------
ds_cfg = ParquetDatasetConfig(
    path=DATASET_PATH,
    clean_data=True,
    target_class=[0, 3, 4, 7, 8],
    event_type=[EventPrefixEnum.REAL],
    split=None,
)
ds = ParquetDataset(ds_cfg)
print(f"[DATA] dataset loaded, total events: {len(ds)}")

# ---------- Windowing configs ----------
wind = Windowing(WindowingConfig(
    window="hann",
    window_size=WINDOW_SIZE,
    overlap=WINDOW_OVERLAP,
    pad_last_window=PAD_LAST
))
label_wind = Windowing(WindowingConfig(
    window="boxcar",
    window_size=WINDOW_SIZE,
    overlap=WINDOW_OVERLAP,
    pad_last_window=PAD_LAST
))

# ---------- Helpers ----------
def n_windows_for_length(L, window_size, overlap, pad_last_window):
    if L <= 0:
        return 0
    step = int(window_size * (1 - overlap))
    if step <= 0:
        step = 1
    if pad_last_window:
        return int(np.ceil(L / step))
    else:
        if L < window_size:
            return 0
        return 1 + (L - window_size) // step

# ---------- SPLIT EVENTS BEFORE WINDOWING ----------
all_indices = np.arange(len(ds))
event_labels = np.array([event["label"]["class"].mode()[0] for event in ds])

# Stratified split at event level (here stratify=None to avoid rare class issues)
idx_trainval, idx_test = train_test_split(
    all_indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=None
)
rel_val = VAL_SIZE / (1 - TEST_SIZE)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=rel_val, random_state=RANDOM_SEED, stratify=None
)

ds_train = [ds[i] for i in idx_train]
ds_val   = [ds[i] for i in idx_val]
ds_test  = [ds[i] for i in idx_test]

print(f"Event split: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")

# ---------- PASS 1: count windows per split ----------
def count_windows(dataset):
    total = 0
    n_signals = None
    for event in dataset:
        sig_df = event["signal"]
        L = len(sig_df)
        total += n_windows_for_length(L, WINDOW_SIZE, WINDOW_OVERLAP, PAD_LAST)
        if n_signals is None:
            n_signals = sig_df.shape[1]
    return total, n_signals

total_train, n_signals = count_windows(ds_train)
total_val, _ = count_windows(ds_val)
total_test, _ = count_windows(ds_test)

print(f"Total windows: train={total_train}, val={total_val}, test={total_test}, signals={n_signals}")

# ---------- Create memmaps ----------
def create_memmaps(prefix, total, n_signals):
    X_seq = np.memmap(TMP_DIR / f"X_{prefix}_seq.dat", dtype=np.float32, mode="w+", shape=(total, WINDOW_SIZE, n_signals))
    X_cnn = np.memmap(TMP_DIR / f"X_{prefix}_cnn.dat", dtype=np.float32, mode="w+", shape=(total, n_signals, WINDOW_SIZE))
    Y     = np.memmap(TMP_DIR / f"Y_{prefix}.dat", dtype=np.int64, mode="w+", shape=(total,))
    return X_seq, X_cnn, Y

X_train_seq, X_train_cnn, y_train = create_memmaps("train", total_train, n_signals)
X_val_seq,   X_val_cnn,   y_val   = create_memmaps("val", total_val, n_signals)
X_test_seq,  X_test_cnn,  y_test  = create_memmaps("test", total_test, n_signals)

# ---------- PASS 2: window events into memmaps ----------
def window_events(dataset, x_seq_mem, x_cnn_mem, y_mem):
    idx = 0
    for event in tqdm(dataset, desc="window events"):
        sig_df = event["signal"]
        labels = event["label"]["class"]

        # window signals
        win_df = wind(sig_df).drop(columns=["win"], errors="ignore")
        arr = win_df.values
        n_w = arr.shape[0]
        if n_w == 0:
            continue
        arr_seq = arr.reshape(n_w, WINDOW_SIZE, sig_df.shape[1])

        # window labels
        lab_win = label_wind(labels).drop(columns=["win"], errors="ignore")
        lab_arr = pd.DataFrame(lab_win.values).mode(axis=1)[0].astype(int).values

        x_seq_mem[idx:idx+n_w] = arr_seq.astype(np.float32)
        x_cnn_mem[idx:idx+n_w] = arr_seq.transpose(0,2,1).astype(np.float32)
        y_mem[idx:idx+n_w] = lab_arr
        idx += n_w
    return idx

filled_train = window_events(ds_train, X_train_seq, X_train_cnn, y_train)
filled_val   = window_events(ds_val, X_val_seq, X_val_cnn, y_val)
filled_test  = window_events(ds_test, X_test_seq, X_test_cnn, y_test)

print(f"Windows written: train={filled_train}, val={filled_val}, test={filled_test}")

# ---------- Prepare MLP data ----------
X_train_mlp = X_train_seq.reshape(filled_train, -1)
X_val_mlp   = X_val_seq.reshape(filled_val, -1)
X_test_mlp  = X_test_seq.reshape(filled_test, -1)


# ---------- Handle class mapping ----------
classes, counts = np.unique(y_train[:filled_train], return_counts=True)
num_classes = len(classes)

if not np.array_equal(classes, np.arange(num_classes)):
    mapping = {old:new for new,old in enumerate(sorted(classes))}
    print("Remapping labels:", mapping)
    y_train[:] = np.array([mapping[int(v)] for v in y_train[:filled_train]], dtype=np.int64)
    y_val[:]   = np.array([mapping[int(v)] for v in y_val[:filled_val]], dtype=np.int64)
    y_test[:]  = np.array([mapping[int(v)] for v in y_test[:filled_test]], dtype=np.int64)
else:
    y_train = y_train[:filled_train]
    y_val   = y_val[:filled_val]
    y_test  = y_test[:filled_test]

# ---------- Train + Evaluate helper ----------
def save_confusion_matrix(cm, classes, path_png: Path, path_csv: Path):
    cm_percent = cm.astype(np.float64)
    row_sums = cm_percent.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_percent, row_sums, out=np.zeros_like(cm_percent), where=row_sums!=0)
    cm_percent *= 100.0
    pd.DataFrame(cm_percent, index=[f"true_{c}" for c in classes],
                 columns=[f"pred_{c}" for c in classes]).to_csv(path_csv)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="viridis",
                vmin=0, vmax=100, xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (%)")
    plt.tight_layout(); plt.savefig(path_png, dpi=300); plt.close()

def train_model_and_eval(name, cfg, cls, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n=== TRAINING {name} ===")
    trainer_cfg = TrainerConfig(
        optimizer="adam", criterion="cross_entropy",
        batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED, device=DEVICE, config_model=cfg,
        cross_validation=False, shuffle_train=True
    )
    trainer = ModelTrainer(trainer_cfg)
    print("Model:\n", trainer.model)

    assessor = ModelAssessment(ModelAssessmentConfig(
        metrics=["balanced_accuracy","precision","recall","f1"],
        task_type=TaskType.CLASSIFICATION,
        class_names=[f"Class_{i}" for i in range(num_classes)],
        export_results=True, generate_report=False
    ))

    t0 = time.perf_counter()
    # ----------------------------------------------------------------------
    # SPECIAL HANDLING FOR MLP (because 3WToolkit MLP.fit() has a different API)
    # ----------------------------------------------------------------------
    if name.lower() == "mlp":
        print("[MLP] Using custom fit() call...")
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # Direct call to the MLP .fit() expected signature
        history = trainer.model.fit(
            train_loader=train_loader,
            epochs=EPOCHS,
            val_loader=val_loader,
            #epochs=EPOCHS,
            #learning_rate=LEARNING_RATE,
            #verbose=True,
            optimizer=optimizer,
            criterion=criterion
        )

    # ----------------------------------------------------------------------
    # NORMAL MODELS (LSTM, GRU, CNN)
    # ----------------------------------------------------------------------
    else:
        trainer.train(
            x_train=X_train,
            y_train=y_train,
            x_val=X_val,
            y_val=y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

    elapsed = time.perf_counter() - t0
    print(f"[{name}] Training time: {elapsed:.2f} seconds")

    # Validation
    print("Validation:")
    val_res = assessor.evaluate(trainer.model, X_val, y_val)
    cm_val = confusion_matrix(val_res["true_values"], val_res["predictions"], labels=np.arange(num_classes))
    save_confusion_matrix(cm_val, np.arange(num_classes), RESULTS_DIR / f"confusion_{name}_val.png", RESULTS_DIR / f"confusion_{name}_val.csv")

    # Test
    print("Test:")
    test_res = assessor.evaluate(trainer.model, X_test, y_test)
    cm_test = confusion_matrix(test_res["true_values"], test_res["predictions"], labels=np.arange(num_classes))
    save_confusion_matrix(cm_test, np.arange(num_classes), RESULTS_DIR / f"confusion_{name}_test.png", RESULTS_DIR / f"confusion_{name}_test.csv")

    # ---- Save Loss Curves ----
    if hasattr(trainer, "history") and len(trainer.history) > 0:
        hist = trainer.history[0]
        train_loss = hist.get("train_loss", [])
        val_loss = hist.get("val_loss", [])

        plt.figure(figsize=(10, 4))
        plt.plot(train_loss, label="Train Loss", marker="o")
        plt.plot(val_loss, label="Val Loss", marker="s")
        plt.title(f"{name} – Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save PNG
        loss_png = RESULTS_DIR / f"loss_curve_{name}.png"
        plt.savefig(loss_png, dpi=300)
        plt.close()

        print(f"[{name}] Loss curve saved to {loss_png}")
    elif name.lower() == "mlp":
        train_loss = history.get("train_loss", []) 
        val_loss = history.get("val_loss", [])

        plt.figure(figsize=(10, 4))
        plt.plot(train_loss, label="Train Loss", marker="o")
        plt.plot(val_loss, label="Val Loss", marker="s")
        plt.title(f"{name} – Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save PNG
        loss_png = RESULTS_DIR / f"loss_curve_{name}.png"
        plt.savefig(loss_png, dpi=300)
        plt.close()

        print(f"[{name}] Loss curve saved to {loss_png}")


    # Save meta
    pd.DataFrame([{
        "model": name,
        "train_time_seconds": elapsed,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "n_test_samples": len(X_test),
        "num_classes": num_classes
    }]).to_csv(RESULTS_DIR / f"meta_{name}.csv", index=False)

    print(f"[{name}] Confusion matrices and meta saved.")
    return trainer, assessor, elapsed

# ---------- Train Models ----------
trained = {}

if RUN_LSTM:
    lstm_cfg = LSTMConfig(input_size=n_signals, hidden_size=64, num_layers=2,
                          output_size=num_classes, random_seed=RANDOM_SEED)
    trained["lstm"] = train_model_and_eval("LSTM", lstm_cfg, LSTM,
                                           X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test)

if RUN_GRU:
    gru_cfg = GRUConfig(input_size=n_signals, hidden_size=64, num_layers=2,
                        output_size=num_classes, random_seed=RANDOM_SEED)
    trained["gru"] = train_model_and_eval("GRU", gru_cfg, GRU,
                                          X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test)

if RUN_CNN:
    cnn_cfg = CNNConfig(input_channels=n_signals, output_size=num_classes,
                        conv_channels=[16,32], kernel_sizes=[3,3],
                        activation_function="relu", random_seed=RANDOM_SEED)
    trained["cnn"] = train_model_and_eval("CNN", cnn_cfg, CNN,
                                          X_train_cnn, y_train, X_val_cnn, y_val, X_test_cnn, y_test)

if RUN_MLP:
    # infer flattened input dim from data and pass to MLPConfig
    mlp_input_size = X_train_mlp.shape[1]
    print(f"[MLP] inferred input size (flattened): {mlp_input_size}")
    mlp_cfg = MLPConfig(
        input_size=mlp_input_size,
        hidden_sizes=(32, 16),
        output_size=num_classes,  
        random_seed=RANDOM_SEED,
        activation_function="relu",
        regularization=0.01,
    )

    trained["mlp"] = train_model_and_eval(
        "MLP", mlp_cfg, MLP,
        X_train_mlp, y_train,
        X_val_mlp, y_val,
        X_test_mlp, y_test
    )



print("\nAll requested models trained successfully. Results saved under:", RESULTS_DIR)
