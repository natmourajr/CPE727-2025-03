"""
Train LeNet on Fashion MNIST with fixed hyperparameters, log to MLflow,
and generate confusion matrix + metrics + copy training curves.

Usage (from repo root):
    uv run python scripts/run_lenet_final.py
"""
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset

from src.config import MLRUNS_DIR
from src.data_loader import FashionMNISTLoader
from src.train_deep import prepare_fashion_mnist_data, train_final_model


def train_lenet_final(params):
    """Train final LeNet model with given params; return run_id and test accuracy."""
    data = prepare_fashion_mnist_data()
    X_train, y_train, X_test, y_test = data["full"]

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    experiment_name = "fashion_mnist_lenet_final"
    mlflow.set_experiment(experiment_name)

    # Train and log inside a single run
    with mlflow.start_run(run_name="lenet_final") as run:
        run_id = run.info.run_id

        acc, classifier = train_final_model(
            X_train,
            y_train,
            X_test,
            y_test,
            dataset="fashion_mnist",
            model_type="lenet",
            params=params,
            experiment_name=None,  # we handle logging here
        )

        # Log params/metrics/model in the active run
        mlflow.log_params({k: v for k, v in params.items() if k != "log_to_mlflow"})
        mlflow.log_metric("test_accuracy", acc)
        try:
            mlflow.pytorch.log_model(classifier.model, "model")
        except Exception as e:
            print(f"Warning: could not log model: {e}")

        return run_id, acc


def eval_and_save_artifacts(run_id):
    """Load logged model, compute metrics, save confusion matrix and copy curves."""
    root = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

    # Load model from run
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    model.eval()

    # Load test set
    loader = FashionMNISTLoader(flatten=False, normalize=True)
    _, _, _, _, X_test, y_test = loader.get_numpy_arrays()
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)

    dl = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=256,
        shuffle=False,
    )

    preds = []
    with torch.no_grad():
        for bx, _ in dl:
            out = model(bx)
            preds.extend(out.argmax(dim=1).cpu().numpy())
    preds = np.array(preds)

    # Metrics
    report = classification_report(y_test, preds, output_dict=True, digits=4)
    acc = report["accuracy"]
    prec = report["macro avg"]["precision"]
    rec = report["macro avg"]["recall"]
    f1 = report["macro avg"]["f1-score"]

    print(
        f"Accuracy: {acc:.4f} | Precision_macro: {prec:.4f} | "
        f"Recall_macro: {rec:.4f} | F1_macro: {f1:.4f}"
    )

    # Save metrics file
    metrics_path = root / "results" / "metrics_fashion_lenet_final.txt"
    metrics_path.write_text(
        "\n".join(
            [
                "Fashion MNIST LeNet Final Metrics",
                f"Accuracy: {acc:.4f}",
                f"Precision_macro: {prec:.4f}",
                f"Recall_macro: {rec:.4f}",
                f"F1_macro: {f1:.4f}",
            ]
        )
        + "\n"
    )
    print(f"Saved metrics to {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=loader.get_class_names()
    ).plot(cmap="Blues", values_format="d").figure_
    fig.set_figwidth(10)
    fig.set_figheight(8)
    fig.tight_layout()

    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "confusion_matrix_lenet_final.png"
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    print(f"Saved confusion matrix to {cm_path}")

    # Copy training curves if present
    tc_src = (
        root
        / "results"
        / "mlruns"
        / run_id[:3]
        / run_id
        / "artifacts"
        / "training_curves.png"
    )
    tc_dst = plots_dir / "training_curves_lenet_final.png"
    if tc_src.exists():
        tc_dst.write_bytes(tc_src.read_bytes())
        print(f"Copied training curves to {tc_dst}")
    else:
        print("Training curves not found to copy.")


def main():
    params = {
        "learning_rate": 0.002,
        "dropout": 0.5,
        "batch_size": 32,
        "epochs": 10,
        "log_to_mlflow": True,
    }
    run_id, test_acc = train_lenet_final(params)
    print(f"Run ID: {run_id} | Test accuracy (logged): {test_acc:.4f}")
    eval_and_save_artifacts(run_id)


if __name__ == "__main__":
    main()

