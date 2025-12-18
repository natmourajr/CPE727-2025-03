"""
Train TextCNN on AG_NEWS with fixed hyperparameters, log to MLflow,
and generate confusion matrix + metrics + copy training curves.

Usage (from repo root):
    uv run python scripts/run_textcnn_final.py
"""
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset

from src.config import MLRUNS_DIR
from src.train_deep import prepare_agnews_data, train_final_model


def train_textcnn_final(params):
    """Train final TextCNN model with given params; return run_id and test accuracy."""
    data = prepare_agnews_data()
    X_train, y_train, X_test, y_test = data["full"]

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    experiment_name = "ag_news_textcnn_final"
    mlflow.set_experiment(experiment_name)

    # Train and log
    with mlflow.start_run(run_name="textcnn_final") as run:
        run_id = run.info.run_id
        acc, _ = train_final_model(
            X_train,
            y_train,
            X_test,
            y_test,
            dataset="ag_news",
            model_type="textcnn",
            params=params,
            experiment_name=None,  # run already opened above
        )
        return run_id, acc


def eval_and_save_artifacts(run_id):
    """Load logged model, compute metrics, save confusion matrix, copy curves."""
    root = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

    # Load model from run
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    model.eval()

    # Load test set
    data = prepare_agnews_data()
    _, _, X_test, y_test = data["full"][0], data["full"][1], data["full"][2], data["full"][3]
    # Actually prepare_agnews_data returns (X_train, y_train, X_test, y_test) as full
    X_train, y_train, X_test, y_test = data["full"]

    dl = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)),
        batch_size=256,
        shuffle=False,
    )

    preds = []
    with torch.no_grad():
        for (bx, _) in dl:
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
    metrics_path = root / "results" / "metrics_agnews_textcnn_final.txt"
    metrics_path.write_text(
        "\n".join(
            [
                "AG News TextCNN Final Metrics",
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
        confusion_matrix=cm, display_labels=["World", "Sports", "Business", "Sci/Tech"]
    ).plot(cmap="Blues", values_format="d").figure_
    fig.set_figwidth(8)
    fig.set_figheight(6)
    fig.tight_layout()

    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm_path = plots_dir / "confusion_matrix_textcnn_final.png"
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    print(f"Saved confusion matrix to {cm_path}")

    # Copy training curves if present
    # training_curves.png is generated during fit when log_to_mlflow=True
    # It is stored under run artifacts
    tc_run = (
        root
        / "results"
        / "mlruns"
        / mlflow.get_run(run_id).info.experiment_id
        / run_id
        / "artifacts"
        / "training_curves.png"
    )
    tc_dst = plots_dir / "training_curves_textcnn_final.png"
    if tc_run.exists():
        tc_dst.write_bytes(tc_run.read_bytes())
        print(f"Copied training curves to {tc_dst}")
    else:
        print("Training curves not found to copy.")


def main():
    params = {
        "learning_rate": 0.001,
        "embedding_dim": 300,
        "num_filters": 100,
        "dropout": 0.3,
        "batch_size": 32,
        "epochs": 5,  # adjust if you want faster (3) or a bit better (5)
        "vocab_size": 10000,
        "log_to_mlflow": True,
    }
    run_id, test_acc = train_textcnn_final(params)
    print(f"Run ID: {run_id} | Test accuracy (logged): {test_acc:.4f}")
    eval_and_save_artifacts(run_id)


if __name__ == "__main__":
    main()

