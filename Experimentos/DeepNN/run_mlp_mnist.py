# run_mlp_mnist.py
# -*- coding: utf-8 -*-
"""
Script de teste para as classes do arquivo mlp_mnist.py:
 - Cria data loaders
 - Treina o MLP
 - Avalia em teste
 - Salva e recarrega o modelo
 - Faz inferência em algumas amostras
"""

from __future__ import annotations
import os
import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib as mpl


from mlp_mnist import data_loader, DataConfig, MLPClassifier


def evaluate(model: MLPClassifier, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    total_acc, total = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(model.device, non_blocking=True)
            targets = targets.to(model.device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            total_acc += (preds == targets).sum().item()
            total += targets.numel()
    model.train()
    return total_acc / max(total, 1)


def main():
    # Carga e preparação dos dados
    cfg = DataConfig(data_dir="./../../Data/DeepNN", batch_size=128, val_split=0.1, num_workers=2, normalize=True)
    dl = data_loader(cfg)
    train_loader, val_loader, test_loader = dl.get_loaders()

    if os.path.exists(cfg.data_dir):
        print(f"Dataset MNIST encontrado em: {cfg.data_dir}")
    else:
        print(f"Dataset MNIST NÃO encontrado em: {cfg.data_dir}. Será baixado...")  

    # Desenvolvimento do Modelo
    model = MLPClassifier(
        input_size=784,
        hidden_sizes=[512, 256, 128],
        output_size=10,
        dropout=0.2,
        seed=42,
    )
    print("Dispositivo:", model.device)

    # Treino
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,           # aumente para melhor acurácia
        lr=1e-3,
        weight_decay=1e-4,
        log_every=100,
    )

    # Avaliação no conjunto de teste
    test_acc = evaluate(model, test_loader)
    print(f"Acurácia em TESTE: {test_acc:.4f}")

    # Plotar curvas de treino/validação
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], 'bo-', label="Train Loss")
    plt.plot(history["val_loss"], 'ro-', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.grid()
    plt.tight_layout()
    plt.show()
    for figformat in ["png", "pdf"]:
        plt.savefig("./../../Data/DeepNN/mlp_mnist_training_curves.%s"%(figformat), dpi=200, bbox_inches='tight')
    plt.close()

    # Salvar e recarregar
    save_path = os.path.join("../../Data/DeepNN/checkpoints", "mlp_mnist.pt")
    model.save_model(save_path)
    loaded = MLPClassifier.load_model(save_path, device=model.device)

    # Predição (probabilidades opcionais)
    y_pred, y_prob = loaded.predict(test_loader, return_probs=True)
    print("Predições obtidas:", y_pred.shape, "| Probabilidades:", y_prob.shape)

    # Plot da matriz de confusão
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    all_targets = np.concatenate([targets.numpy() for _, targets in test_loader], axis=0)
    disp = ConfusionMatrixDisplay.from_predictions(all_targets, y_pred, normalize='true', cmap='Blues', values_format=".2f", colorbar=True)
    disp.ax_.set_title("Matriz de Confusão (normalizada %)")
    plt.tight_layout()
    plt.show()
    for figformat in ["png", "pdf"]:
        plt.savefig("./../../Data/DeepNN/mlp_mnist_confusion_matrix.%s"%(figformat), dpi=200, bbox_inches='tight')
    plt.close() 

    # Visualizar algumas amostras e suas predições
    #    (opcional — útil para verificar o pipeline)
    batch_images, batch_targets = next(iter(test_loader))
    with torch.no_grad():
        logits = loaded(batch_images.to(loaded.device))
        preds = torch.argmax(logits, dim=1).cpu()

    grid = make_grid(batch_images[:16], nrow=8, pad_value=1.0)
    plt.figure(figsize=(10, 3))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    title = "Predições (topo-esq →): " + " ".join(map(str, preds[:16].tolist()))
    plt.title(title)
    plt.tight_layout()
    plt.show()
    for figformat in ["png", "pdf"]:
        plt.savefig("./../../Data/DeepNN/mlp_mnist_predictions.%s"%(figformat), dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
