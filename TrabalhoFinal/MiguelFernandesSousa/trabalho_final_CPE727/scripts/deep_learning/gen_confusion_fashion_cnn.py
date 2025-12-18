"""
Generate confusion matrix for the Fashion-MNIST CNN (1-epoch run) and log it to MLflow.

Usage (from repo root):
    uv run python scripts/gen_confusion_fashion_cnn.py
"""
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset

from src.models_deep import FashionMNISTCNN
from src.data_loader import FashionMNISTLoader


def load_model(model_path: Path, dropout: float) -> FashionMNISTCNN:
    """
    Load a FashionMNISTCNN from a saved artifact. The artifact may store either
    the full model object or a state_dict.
    """
    loaded = torch.load(model_path, map_location="cpu")

    if isinstance(loaded, FashionMNISTCNN):
        model = loaded
    elif isinstance(loaded, torch.nn.Module):
        # Just in case it was saved as a different module type
        model = loaded
    elif isinstance(loaded, dict):
        model = FashionMNISTCNN(dropout=dropout)
        model.load_state_dict(loaded)
    else:
        raise TypeError(f"Unsupported model artifact type: {type(loaded)}")

    model.eval()
    return model


def build_test_loader(batch_size: int = 128):
    loader = FashionMNISTLoader(flatten=False, normalize=True)
    _, _, _, _, X_test, y_test = loader.get_numpy_arrays()
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)

    tensor_x = torch.tensor(X_test, dtype=torch.float32)
    tensor_y = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), loader.get_class_names()


def compute_predictions(model: FashionMNISTCNN, dataloader: DataLoader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            y_true.extend(batch_y.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    # Run metadata (1-epoch CNN)
    tracking_uri = "results/mlruns"
    run_id = "6ca1072260884e3bbac2ee937ad452a2"
    model_artifact = Path(tracking_uri) / "592721634892349474" / "models" / "m-9f91f4a975d44693b19ac27e8eaf2032" / "artifacts" / "data" / "model.pth"
    dropout = 0.3  # recorded in run params

    assert model_artifact.exists(), f"Model artifact not found: {model_artifact}"

    # Load model and data
    model = load_model(model_artifact, dropout=dropout)
    test_loader, class_names = build_test_loader(batch_size=128)

    # Predictions
    y_true, y_pred = compute_predictions(model, test_loader)
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig = disp.plot(include_values=True, cmap="Blues", xticks_rotation=45).figure_

    # Save locally
    out_path = Path("results/plots/confusion_matrix_cnn_1epoch.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix to {out_path}")

    # Log to MLflow run
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_figure(fig, "confusion_matrix_test.png")
    print(f"Logged confusion matrix to MLflow run {run_id}")


if __name__ == "__main__":
    main()

