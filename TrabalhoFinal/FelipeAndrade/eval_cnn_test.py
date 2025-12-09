import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from datasets.mbgv2_crops_dataset import Mbgv2CropsDataset
from models.cnn_classifier import SimpleCNN


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)

    return acc, f1, cm, all_targets, all_preds


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "cnn_base")
    ckpt_path = os.path.join(results_dir, "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Transform igual ao de validação
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    test_ds = Mbgv2CropsDataset(split="test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    num_classes = len(test_ds.classes)
    model = SimpleCNN(num_classes=num_classes).to(device)

    print("Carregando checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint treinado até epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

    acc, f1, cm, y_true, y_pred = evaluate_test(model, test_loader, device)
    print("\n=== Resultados no conjunto de teste ===")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Macro-F1:  {f1:.4f}")

    print("\nMatriz de confusão (linhas = verdade, colunas = predição):")
    print(cm)

    print("\nRelatório por classe:")
    print(classification_report(y_true, y_pred, target_names=test_ds.classes))


if __name__ == "__main__":
    main()
