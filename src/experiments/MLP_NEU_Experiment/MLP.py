import sys, os, json, copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import seaborn as sns
import pandas as pd

from src.models.MLP_NEU.model import MLP_NEU
from src.dataloaders.NEU_loader.loader import NEUDataset, default_transform



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(dataloader, desc="Treinando", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        imgs = imgs.view(imgs.size(0), -1)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            imgs = imgs.view(imgs.size(0), -1)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    acc = 100.0 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, precision, recall, f1, cm


def plot_loss_and_f1(history, save_path_prefix, title_prefix=""):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Treino", marker="o")
    plt.plot(epochs, history["val_loss"], label="Validação", marker="o")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix}Loss")
    plt.legend()

    # F1 de validação
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_f1"], label="F1 Validação", marker="o")
    plt.xlabel("Época")
    plt.ylabel("F1-score (macro)")
    plt.title(f"{title_prefix}F1 na Validação")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path_prefix + "_loss_f1.png")
    plt.close()


def summarize_uncertainty(values, name):
    """Retorna média, desvio e IC ~95% (normal) para uma métrica."""
    values = np.array(values, dtype=float)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # desvio amostral
    n = len(values)
    se = std / np.sqrt(n)
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    print(f"{name}: média={mean:.4f}, desvio={std:.4f}, "
          f"IC95%≈[{ci_low:.4f}, {ci_high:.4f}]")
    return mean, std, ci_low, ci_high



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    torch.backends.cudnn.benchmark = True

    root_dir = os.path.join("Data", "NEU-DET")
    save_dir = os.path.join("src", "Results", "MLP_NEU")
    os.makedirs(save_dir, exist_ok=True)

    class_names = ["Crazing", "Inclusion", "Patches",
                   "Pitted_surface", "Rolled_in_scale", "Scratches"]

    transform = default_transform()
    dataset = NEUDataset(root_dir=root_dir, transform=transform)
    n_total = len(dataset)
    labels = [label for _, label in dataset.samples]
    print(f"[Dataset] Total de amostras: {n_total}")

    test_ratio = 0.15
    seed = 42
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss.split(np.arange(n_total), labels))

    trainval_idx = np.array(trainval_idx)
    test_idx = np.array(test_idx)
    trainval_labels = [labels[i] for i in trainval_idx]

    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)

    print(f"[Split] Train+Val: {len(trainval_idx)} | Test (holdout): {len(test_idx)}")

    k_folds = 10
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    max_epochs = 100
    patience = 10
    criterion = nn.CrossEntropyLoss()

    metrics_summary = []  # métricas de validação por fold
    best_fold = -1
    best_f1_global = -1.0
    best_model_state = None
    best_history = None

    for fold_idx, (train_local_idx, val_local_idx) in enumerate(
        skf.split(trainval_idx, trainval_labels)
    ):
        print(f"\n Iniciando Fold {fold_idx+1}/{k_folds}")

        train_indices = trainval_idx[train_local_idx]
        val_indices = trainval_idx[val_local_idx]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        model = MLP_NEU().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": []
        }

        best_f1_fold = -1.0
        best_state_fold = None
        patience_counter = 0

        for epoch in range(max_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(f"[Fold {fold_idx+1}] Época {epoch+1}/{max_epochs} | "
                  f"Treino: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                  f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}% | "
                  f"F1={val_f1*100:.2f}%")

            if val_f1 > best_f1_fold + 1e-4:
                best_f1_fold = val_f1
                best_state_fold = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Fold {fold_idx+1}] Early stopping na época {epoch+1}")
                    break

        best_model_fold = MLP_NEU().to(device)
        best_model_fold.load_state_dict(best_state_fold)
        _, val_acc_final, val_prec_final, val_rec_final, val_f1_final, _ = evaluate(
            best_model_fold, val_loader, criterion, device
        )

        metrics_summary.append(
            (val_acc_final, val_prec_final, val_rec_final, val_f1_final)
        )

        if val_f1_final > best_f1_global:
            best_f1_global = val_f1_final
            best_fold = fold_idx + 1
            best_model_state = copy.deepcopy(best_state_fold)
            best_history = history

    accs, precs, recs, f1s = zip(*metrics_summary)

    print("\n Resumo K-Fold (10 folds) com incertezas (validação):")
    acc_stats = summarize_uncertainty(accs, "Accuracy")
    prec_stats = summarize_uncertainty(precs, "Precision")
    rec_stats = summarize_uncertainty(recs, "Recall")
    f1_stats = summarize_uncertainty(f1s, "F1-score")

    df_folds = pd.DataFrame(
        {
            "Fold": list(range(1, k_folds + 1)),
            "Accuracy": accs,
            "Precision": precs,
            "Recall": recs,
            "F1": f1s,
        }
    )
    df_folds.to_csv(os.path.join(save_dir, "metrics_kfold_folds.csv"), index=False)

    df_unc = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Mean": [acc_stats[0], prec_stats[0], rec_stats[0], f1_stats[0]],
            "Std": [acc_stats[1], prec_stats[1], rec_stats[1], f1_stats[1]],
            "CI95_low": [acc_stats[2], prec_stats[2], rec_stats[2], f1_stats[2]],
            "CI95_high": [acc_stats[3], prec_stats[3], rec_stats[3], f1_stats[3]],
        }
    )
    df_unc.to_csv(os.path.join(save_dir, "metrics_kfold_uncertainty.csv"), index=False)

    info = {
        "best_fold": best_fold,
        "best_f1_val": float(best_f1_global),
    }
    with open(os.path.join(save_dir, "best_fold_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n🏆 Melhor Fold: {best_fold} com F1_val={best_f1_global*100:.2f}%")

    plot_loss_and_f1(
        best_history,
        save_path_prefix=os.path.join(save_dir, f"best_fold{best_fold}"),
        title_prefix=f"MLP - Fold {best_fold} - "
    )

    print("\n Avaliando o melhor fold no conjunto de TESTE (holdout externo)...")
    best_model = MLP_NEU().to(device)
    best_model.load_state_dict(best_model_state)

    test_loss, test_acc, test_prec, test_rec, test_f1, cm_test = evaluate(
        best_model, test_loader, criterion, device
    )

    print("\n Resultados no TESTE (Holdout):")
    print(f"Accuracy = {test_acc:.2f}%")
    print(f"Precision (macro) = {test_prec*100:.2f}%")
    print(f"Recall (macro) = {test_rec*100:.2f}%")
    print(f"F1-score (macro) = {test_f1*100:.2f}%")

    # Matriz de confusão  
    df_cm = pd.DataFrame(cm_test, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão - TESTE (Holdout externo) - MLP")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_test_holdout.png"))
    plt.close()

    df_test = pd.DataFrame(
        [{
            "Accuracy": test_acc,
            "Precision": test_prec,
            "Recall": test_rec,
            "F1": test_f1,
        }]
    )
    df_test.to_csv(os.path.join(save_dir, "metrics_test_holdout.csv"), index=False)

    print(f"\n Resultados K-Fold + Holdout externo salvos em: {save_dir}")


if __name__ == "__main__":
    main()
