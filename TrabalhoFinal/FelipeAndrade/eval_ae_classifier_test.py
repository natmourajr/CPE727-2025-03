import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from models.ae_mlp_classifier import AEMLPClassifier


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_features_split(results_dir: str, split: str):
    path = os.path.join(results_dir, f"features_{split}.pt")
    data = torch.load(path)
    features = data["features"].float()  # [N, 256]
    labels = data["labels"].long()      # [N]
    classes = data["classes"]
    return features, labels, classes


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    for feats, labels in loader:
        feats = feats.to(device)
        labels = labels.to(device)

        outputs = model(feats)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)

    return acc, f1, cm, all_targets, all_preds


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "autoencoder")
    clf_dir = os.path.join(results_dir, "classifier")
    ckpt_path = os.path.join(clf_dir, "best_classifier.pt")

    device = get_device()
    print("Device:", device)

    X_test, y_test, classes = load_features_split(results_dir, "test")
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    in_dim = X_test.shape[1]
    num_classes = len(classes)
    print("Num classes:", num_classes)
    print("Input feature dim:", in_dim)

    model = AEMLPClassifier(in_dim=in_dim, num_classes=num_classes, hidden_dim=256, dropout_p=0.3).to(device)

    print("Carregando checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint treinado até epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

    acc, f1, cm, y_true, y_pred = evaluate_test(model, test_loader, device)

    print("\n=== Resultados no conjunto de teste (AE + MLP) ===")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Macro-F1:  {f1:.4f}")

    print("\nMatriz de confusão (linhas = verdade, colunas = predição):")
    print(cm)

    print("\nRelatório por classe:")
    print(classification_report(y_true, y_pred, target_names=classes))


if __name__ == "__main__":
    main()
