# Pure GRU Training with K-Fold
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Toolkit imports (dataset + windowing) ---
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../3WToolkit/')))
from ThreeWToolkit.dataset import ParquetDataset
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig, EventPrefixEnum
from ThreeWToolkit.core.base_preprocessing import WindowingConfig
from ThreeWToolkit.preprocessing import Windowing

# ============================================================
# 1) PURE GRU
# ============================================================
class Pure_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,
                 dropout=0.3, lr=1e-4, weight_decay=1e-5,
                 num_epochs=100, class_weights=None):
        super(Pure_GRU, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Training parameters
        self.num_epochs = num_epochs
        self.lossList = []
        self.valLossList = []
        
        # Loss function with class weights
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                cw = class_weights.to(torch.float32)
            else:
                cw = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def forward(self, x):
        # x.shape: (batch_size, sequence_length, input_dim)
        gru_out, _ = self.gru(x)
        
        # Take the output from the last timestep
        output = gru_out[:, -1, :]
        return self.fc(output)
    
    def fit(self, train_loader, val_loader, device):
        # Clear lists
        self.lossList = []
        self.valLossList = []
        
        for epoch in range(self.num_epochs):
            # Training
            self.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / max(1, len(train_loader))
            self.lossList.append(train_loss)
            
            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = self.forward(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= max(1, len(val_loader))
            self.valLossList.append(val_loss)
            
            # Update scheduler
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
    
    def predict(self, loader, device):
        self.eval()
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                outputs = self.forward(X_batch)
                _, preds = torch.max(outputs, 1)
                y_pred.extend(preds.cpu().numpy())
        return np.array(y_pred)
    
    def evaluate(self, loader, device):
        self.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)
                running_loss += loss.item()
        return running_loss / max(1, len(loader))


# ============================================================
# 2) DATASET LOADING AND WINDOWING
# ============================================================
RANDOM_SEED = 2025
WINDOW_SIZE = 1000
SELECTED_COLS = ["P-MON-CKP", "T-TPT", "P-TPT", "P-PDG", "P-JUS-CKGL"]

event_types = [EventPrefixEnum.REAL]
ds_config = ParquetDatasetConfig(
    path="./dataset",
    clean_data=True,
    seed=RANDOM_SEED,
    event_type=event_types
)
ds = ParquetDataset(ds_config)
print(f"Dataset loaded. Size: {len(ds)}")

windowing_config = WindowingConfig(
    window="hann",
    window_size=WINDOW_SIZE,
    overlap=0.5,
    pad_last_window=True
)
windowing = Windowing(windowing_config)

dfs = []
print("Applying windowing...")
for event in ds:
    if all(col in event["signal"].columns for col in SELECTED_COLS):
        signal_subset = event["signal"][SELECTED_COLS]
        windowed_signal = windowing(signal_subset)
        if "win" in windowed_signal.columns:
            windowed_signal = windowed_signal.drop(columns=["win"])
        windowed_signal["label"] = np.unique(event["label"]["class"])[0]
        dfs.append(windowed_signal)

if len(dfs) == 0:
    raise RuntimeError("No windows were generated. Check SELECTED_COLS and the dataset.")

dfs_final = pd.concat(dfs, ignore_index=True)
print("Windowing completed. Total windows:", len(dfs_final))

# Prepare X and y
X_data = dfs_final.iloc[:, :-1].values
y_data = dfs_final["label"].astype(int).values

# ============================================================
# 3) RESHAPE FOR GRU (3D: batch, sequence, features)
# ============================================================
sequence_length = WINDOW_SIZE
input_dim = len(SELECTED_COLS)

num_samples = len(X_data)
features_per_window = X_data.shape[1]

# Check reshape
expected_features = sequence_length * input_dim
if features_per_window != expected_features:
    print(f"Warning: features per window ({features_per_window}) != expected ({expected_features})")
    print("Adjusting sequence_length...")
    sequence_length = features_per_window // input_dim

# Reshape
X_data_reshaped = X_data.reshape(num_samples, sequence_length, input_dim)

print(f"Original shape: {X_data.shape}")
print(f"Shape for GRU: {X_data_reshaped.shape}")
print(f"  - Number of samples: {num_samples}")
print(f"  - Sequence length: {sequence_length}")
print(f"  - Input dimension (features): {input_dim}")

# ================================================
# 4) CROSS-VALIDATION LOOP (K-Fold)
# ================================================
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
batch_size = 32
num_epochs = 100

# Results storage
all_val_reports = []
all_val_cms = []
all_val_losses_final = []
all_train_losses_full = []
all_val_losses_full = []
all_val_accuracies = []
all_val_f1_scores = []

# Directory to save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir_base = f"./results/gru_pure/{timestamp}"
os.makedirs(save_dir_base, exist_ok=True)

# Convert to PyTorch tensors
X_data_tensor = torch.tensor(X_data_reshaped, dtype=torch.float32)
y_data_tensor = torch.tensor(y_data, dtype=torch.long)
dataset = TensorDataset(X_data_tensor, y_data_tensor)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

n_classes = len(np.unique(y_data))

# Start K-Fold loop
for fold, (train_index, val_index) in enumerate(kf.split(X_data_reshaped)):
    print(f"\n--- Starting Fold {fold + 1}/{n_splits} ---")
    
    # Subsets and dataloaders
    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Class weights
    y_train_fold = y_data_tensor[train_index].numpy()
    class_counts_fold = np.array([np.sum(y_train_fold == i) for i in range(n_classes)], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse = np.where(class_counts_fold > 0, 1.0 / class_counts_fold, 0.0)
    class_weights_fold = inverse / inverse.sum() if inverse.sum() > 0 else np.ones(n_classes) / n_classes
    class_weights_fold = torch.tensor(class_weights_fold, dtype=torch.float32)
    
    # Instantiate Pure GRU model
    model = Pure_GRU(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=n_classes,
        dropout=0.001,
        lr=1e-4,
        weight_decay=1e-5,
        num_epochs=num_epochs,
        class_weights=class_weights_fold
    ).to(device)
    
    # Train
    model.fit(train_loader, val_loader, device)
    
    # Collect curves
    fold_train_losses = model.lossList.copy()
    fold_val_losses = model.valLossList.copy()
    
    # Save fold curves
    plt.figure(figsize=(10,6))
    if len(fold_train_losses) > 0:
        plt.plot(fold_train_losses, label="Training")
    if len(fold_val_losses) > 0:
        plt.plot(fold_val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - Fold {fold+1}")
    plt.grid(alpha=0.5)
    plt.legend()
    curve_path = os.path.join(save_dir_base, f"loss_curve_fold_{fold+1}.png")
    plt.savefig(curve_path, bbox_inches='tight')
    plt.close()
    print(f"Loss curve for Fold {fold+1} saved at: {curve_path}")
    
    # Final evaluation
    model.eval()
    y_pred_fold = []
    y_true_fold = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred_fold.extend(preds)
            y_true_fold.extend(yb.cpu().numpy())
    
    # Metrics
    accuracy_fold = accuracy_score(y_true_fold, y_pred_fold)
    f1_fold = f1_score(y_true_fold, y_pred_fold, average='weighted')
    report_fold = classification_report(y_true_fold, y_pred_fold, labels=range(n_classes), 
                                       zero_division=0, output_dict=True)
    cm_fold = confusion_matrix(y_true_fold, y_pred_fold, labels=range(n_classes))
    
    # Store results
    all_val_reports.append(report_fold)
    all_val_cms.append(cm_fold)
    all_val_accuracies.append(accuracy_fold)
    all_val_f1_scores.append(f1_fold)
    all_val_losses_final.append(fold_val_losses[-1] if len(fold_val_losses) > 0 else np.nan)
    all_train_losses_full.append(fold_train_losses)
    all_val_losses_full.append(fold_val_losses)
    
    # Print fold summary
    print(f"\n===== Classification Report - Fold {fold+1} =====")
    print(f"Accuracy: {accuracy_fold:.4f}")
    print(f"F1 Score: {f1_fold:.4f}")
    print(classification_report(y_true_fold, y_pred_fold, labels=range(n_classes), zero_division=0))
    print(f"===== Confusion Matrix - Fold {fold+1} =====")
    print(cm_fold)

# --- Post-processing: final statistics ---
mean_accuracy = np.nanmean(all_val_accuracies)
std_accuracy = np.nanstd(all_val_accuracies)
mean_f1 = np.nanmean(all_val_f1_scores)
std_f1 = np.nanstd(all_val_f1_scores)
mean_val_loss = np.nanmean(all_val_losses_final)
std_val_loss = np.nanstd(all_val_losses_final)

print("\n--- Cross-Validation Final Results ---")
print(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Average Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
print(f"Accuracies per Fold: {[f'{acc:.4f}' for acc in all_val_accuracies]}")
print(f"F1 Scores per Fold: {[f'{f1:.4f}' for f1 in all_val_f1_scores]}")

# F1 per fold plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_splits + 1), all_val_f1_scores, alpha=0.7)
plt.axhline(y=mean_f1, color='red', linestyle='--', label=f'Average: {mean_f1:.4f}')
plt.title('F1 Score per Fold - Cross-Validation (Pure GRU)')
plt.xlabel('Fold')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, n_splits + 1))
f1_bar_path = os.path.join(save_dir_base, "f1_score_per_fold.png")
plt.savefig(f1_bar_path, bbox_inches='tight')
plt.close()
print(f"F1 Score per fold plot saved at: {f1_bar_path}")

# Average confusion matrix + deviation
all_val_cms_array = np.array(all_val_cms)
mean_cm = np.mean(all_val_cms_array, axis=0)
std_cm = np.std(all_val_cms_array, axis=0)
cm_labels = np.array([[f'{mean_cm[i, j]:.2f} ± {std_cm[i, j]:.2f}' 
                       for j in range(n_classes)] for i in range(n_classes)])
plt.figure(figsize=(10, 8))
sns.heatmap(mean_cm, annot=cm_labels, fmt='', cmap='Blues',
            xticklabels=[f'Class {i}' for i in range(n_classes)],
            yticklabels=[f'Class {i}' for i in range(n_classes)])
plt.title("Average Confusion Matrix with Standard Deviation (Pure GRU)")
plt.xlabel("Predicted")
plt.ylabel("True")
conf_path_avg_std = os.path.join(save_dir_base, "confusion_matrix_average_std.png")
plt.savefig(conf_path_avg_std, bbox_inches='tight')
plt.close()
print(f"Average confusion matrix with standard deviation saved at: {conf_path_avg_std}")

# Normalized average confusion matrix + deviation (ADJUSTED)
normalized_cms = []
for cm in all_val_cms:
    with np.errstate(invalid='ignore', divide='ignore'):
        row_sums = cm.sum(axis=1)
        norm = np.divide(cm.astype('float'), row_sums[:, np.newaxis], 
                        where=row_sums[:, np.newaxis]!=0)
    normalized_cms.append(norm)
normalized_cms_array = np.array(normalized_cms)
mean_normalized_cm = np.nanmean(normalized_cms_array, axis=0)
std_normalized_cm = np.nanstd(normalized_cms_array, axis=0)

# Version with only average values (more readable)
normalized_cm_labels_simple = np.array([[f'{mean_normalized_cm[i, j]:.1%}' 
                                        for j in range(n_classes)] for i in range(n_classes)])
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(mean_normalized_cm, annot=normalized_cm_labels_simple, fmt='', 
            cmap='Blues', ax=ax, cbar_kws={'label': 'Proportion'},
            xticklabels=[f'Class {i}' for i in range(n_classes)],
            yticklabels=[f'Class {i}' for i in range(n_classes)],
            annot_kws={'size': 14})
plt.title("Normalized Confusion Matrix - Average (Pure GRU)", fontsize=14, pad=15)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.tight_layout()
conf_path_normalized = os.path.join(save_dir_base, "confusion_matrix_normalized_average.png")
plt.savefig(conf_path_normalized, bbox_inches='tight', dpi=300)
plt.close()
print(f"Normalized confusion matrix (average) saved at: {conf_path_normalized}")

# Version with average and deviation in more compact format
normalized_cm_labels_compact = np.array([[f'{mean_normalized_cm[i, j]:.1%}\n±{std_normalized_cm[i, j]:.1%}' 
                                         for j in range(n_classes)] for i in range(n_classes)])
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(mean_normalized_cm, annot=normalized_cm_labels_compact, fmt='', 
            cmap='Blues', ax=ax, cbar_kws={'label': 'Proportion'},
            xticklabels=[f'Class {i}' for i in range(n_classes)],
            yticklabels=[f'Class {i}' for i in range(n_classes)],
            annot_kws={'size': 14})
plt.title("Normalized Confusion Matrix - Average ± Standard Deviation (Pure GRU)", fontsize=14, pad=15)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.tight_layout()
conf_path_normalized_avg_std = os.path.join(save_dir_base, "confusion_matrix_normalized_average_std.png")
plt.savefig(conf_path_normalized_avg_std, bbox_inches='tight', dpi=300)
plt.close()
print(f"Normalized confusion matrix (average ± deviation) saved at: {conf_path_normalized_avg_std}")

# Best fold curve
best_fold_index = int(np.nanargmin(all_val_losses_final))
best_fold_loss = all_val_losses_final[best_fold_index]
best_fold_f1 = all_val_f1_scores[best_fold_index]
best_train_losses = all_train_losses_full[best_fold_index]
best_val_losses = all_val_losses_full[best_fold_index]

print(f"\nBest fold: {best_fold_index + 1}")
print(f"Final validation loss: {best_fold_loss:.4f}")
print(f"F1 Score: {best_fold_f1:.4f}")

plt.figure(figsize=(10, 6))
if len(best_train_losses) > 0:
    plt.plot(best_train_losses, label=f"Training (Best Fold)")
if len(best_val_losses) > 0:
    plt.plot(best_val_losses, label=f"Validation (Best Fold)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss Curve of Best Fold ({best_fold_index + 1}) - F1: {best_fold_f1:.4f}")
plt.grid(alpha=0.5)
plt.legend()
best_curve_path = os.path.join(save_dir_base, "loss_curve_best_fold.png")
plt.savefig(best_curve_path, bbox_inches='tight')
plt.close()
print(f"Loss curve of best fold saved at: {best_curve_path}")

# Save final summary
results_file = os.path.join(save_dir_base, "final_results.txt")
with open(results_file, 'w') as f:
    f.write("=== CROSS-VALIDATION RESULTS (PURE GRU) ===\n\n")
    f.write(f"Number of Folds: {n_splits}\n\n")
    f.write(f"Architecture: Pure GRU\n")
    f.write(f"  GRU Layers:\n")
    f.write(f"    - Input dim (features): {input_dim}\n")
    f.write(f"    - Hidden dim: 64\n")
    f.write(f"    - Num layers: 2\n")
    f.write(f"    - Dropout: 0.3 (between layers)\n")
    f.write(f"    - Output: last temporal output (64,)\n")
    f.write(f"  Classification Layer:\n")
    f.write(f"    - Linear: 64 → {n_classes}\n")
    f.write(f"  Input:\n")
    f.write(f"    - Sequence length: {sequence_length}\n")
    f.write(f"    - Batch size: {batch_size}\n\n")
    f.write(f"Hyperparameters:\n")
    f.write(f"  - Learning rate: 1e-4\n")
    f.write(f"  - Weight decay: 1e-5\n")
    f.write(f"  - Optimizer: AdamW\n")
    f.write(f"  - Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)\n")
    f.write(f"  - Epochs: {num_epochs}\n")
    f.write(f"  - Class weights: Yes (balanced per fold)\n")
    f.write(f"  - Gradient clipping: max_norm=1.0\n\n")
    f.write(f"GRU vs LSTM differences:\n")
    f.write(f"  - GRU has fewer parameters (no cell state)\n")
    f.write(f"  - Training ~25% faster\n")
    f.write(f"  - Performance generally similar\n\n")
    f.write(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
    f.write(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n")
    f.write(f"Average Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}\n")
    f.write(f"Best Fold: {best_fold_index + 1} (F1 Score: {best_fold_f1:.4f})\n\n")
    f.write("Accuracies per Fold:\n")
    for i, acc in enumerate(all_val_accuracies, 1):
        f.write(f"Fold {i}: {acc:.4f}\n")
    f.write("\nF1 Scores per Fold:\n")
    for i, f1_score_val in enumerate(all_val_f1_scores, 1):
        f.write(f"Fold {i}: {f1_score_val:.4f}\n")
print(f"Final results saved at: {results_file}")