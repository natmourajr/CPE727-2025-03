"""Evaluation nodes with MLFlow integration."""
import json
import torch
import mlflow
import logging
import tempfile
import matplotlib
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend

from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple
from eurosat_dl_comparison.pipelines.model_training.models import create_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix



logger = logging.getLogger(__name__)


class DatasetWithTransform(torch.utils.data.Dataset):
    """Wrapper to apply transforms to a dataset."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Apply our transform
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


def load_best_models_config(best_models_csv_path: str) -> pd.DataFrame:
    """Load the best models configuration from CSV."""
    return pd.read_csv(best_models_csv_path)


def get_hyperparameters_from_mlflow(run_id: str, mlflow_tracking_uri: str, experiment_id: str) -> Dict[str, Any]:
    """Recover hyperparameters from MLflow run."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    run = client.get_run(run_id)

    params = run.data.params

    # Convert string parameters to appropriate types
    hyperparams = {}
    for key, value in params.items():
        # Convert boolean strings
        if value.lower() in ['true', 'false']:
            hyperparams[key] = value.lower() == 'true'
        # Try to parse list (e.g., '[512, 256]')
        elif value.startswith('[') and value.endswith(']'):
            try:
                hyperparams[key] = json.loads(value)
            except json.JSONDecodeError:
                hyperparams[key] = value
        # Try to convert to float
        elif '.' in value or 'e' in value.lower():
            try:
                hyperparams[key] = float(value)
            except ValueError:
                hyperparams[key] = value
        # Try to convert to int
        else:
            try:
                hyperparams[key] = int(value)
            except ValueError:
                hyperparams[key] = value

    return hyperparams


def evaluate_best_model(
    model_name: str,
    config_run_id: str,
    dev_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    model_params: Dict[str, Any],
    training_params: Dict[str, Any],
    data_loader_params: Dict[str, Any],
    mlflow_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train the best model on full dev dataset and evaluate on test dataset.

    Args:
        model_name: Name of the model (mlp, vgg16, resnet50, vit_b_16)
        config_run_id: MLflow run ID for the best configuration
        dev_dataset: Full development dataset for training
        test_dataset: Test dataset for evaluation
        model_params: Model parameters from config
        training_params: Training parameters
        data_loader_params: Data loader parameters
        mlflow_params: MLflow parameters

    Returns:
        Dictionary with evaluation results
    """
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])

    # Use separate experiment for final evaluation
    evaluation_experiment_name = mlflow_params.get(
        "evaluation_experiment_name",
        f"{mlflow_params['experiment_name']}_final_evaluation"
    )
    mlflow.set_experiment(evaluation_experiment_name)

    # Get hyperparameters from the best config run
    experiment_id = mlflow_params.get("experiment_id", "914356777944626268")
    hyperparams = get_hyperparameters_from_mlflow(
        config_run_id,
        mlflow_params["tracking_uri"],
        experiment_id
    )

    logger.info(f"Evaluating {model_name} with hyperparams from run {config_run_id}: {hyperparams}")

    # Define transforms (same as training pipeline)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        normalize,
    ])

    # Apply transforms to datasets
    train_dataset = DatasetWithTransform(dev_dataset, train_transform)
    test_dataset_transformed = DatasetWithTransform(test_dataset, eval_transform)

    # Create data loaders
    batch_size = hyperparams.get("batch_size", data_loader_params["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_params["num_workers"],
        pin_memory=data_loader_params["pin_memory"]
    )
    test_loader = DataLoader(
        test_dataset_transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_params["num_workers"],
        pin_memory=data_loader_params["pin_memory"]
    )

    # Start MLflow run for final evaluation
    with mlflow.start_run(run_name=f"{model_name}_final_evaluation"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("config_run_id", config_run_id)
        # Log dataset sizes safely
        try:
            mlflow.log_param("train_samples", len(dev_dataset))  # type: ignore
            mlflow.log_param("test_samples", len(test_dataset))  # type: ignore
        except TypeError:
            pass  # Dataset doesn't support len()

        # Log hyperparameters
        for key, value in hyperparams.items():
            mlflow.log_param(key, value)

        # Create model with hyperparameters
        model_config = {**model_params}
        model_config.pop("gridsearch", None)

        # Update model config with hyperparameters from best run
        for key, value in hyperparams.items():
            if key not in ["fold", "batch_size", "learning_rate"]:
                model_config[key] = value

        model = create_model(model_name, model_config)
        device = torch.device(training_params["device"] if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Log dataset info
        try:
            num_train_samples = len(dev_dataset)  # type: ignore
            logger.info(f"Training {model_name} on full dev dataset with {num_train_samples} samples")
        except TypeError:
            logger.info(f"Training {model_name} on full dev dataset")

        # Setup optimizer and loss
        learning_rate = hyperparams.get("learning_rate", 0.001)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=training_params["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_train_loss = float('inf')
        best_model_state = None

        epoch_pbar = tqdm(range(training_params["epochs"]), desc=f"Training {model_name}", unit="epoch")
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0

            train_batch_pbar = tqdm(train_loader, desc="Train", leave=False, unit="batch")
            for data, target in train_batch_pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_batch_pbar.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                best_model_state = model.state_dict().copy()

            epoch_pbar.set_postfix(train_loss=avg_train_loss, best_loss=best_train_loss)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}")
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        logger.info(f"Final train_loss: {best_train_loss:.4f}")
        mlflow.log_metric("final_train_loss", best_train_loss)

        # Save model to MLflow
        model_path = Path(tempfile.gettempdir()) / f"{model_name}_final_model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        model_path.unlink()

        # Log dataset info
        try:
            num_test_samples = len(test_dataset)  # type: ignore
            logger.info(f"Evaluating {model_name} on test dataset with {num_test_samples} samples")
        except TypeError:
            logger.info(f"Evaluating {model_name} on test dataset")

        # Evaluate on test dataset
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            test_batch_pbar = tqdm(test_loader, desc=f"Evaluating {model_name}", unit="batch")
            for data, target in test_batch_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()

                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                test_batch_pbar.set_postfix(loss=loss.item())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )

        # Log test metrics
        test_metrics = {
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        logger.info(f"{model_name} Test Results - Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}, Loss: {avg_test_loss:.4f}")

        # Save detailed metrics
        metrics_file = Path(tempfile.gettempdir()) / f"{model_name}_test_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_file), artifact_path="metrics")
        metrics_file.unlink()

        # Save classification report
        report = classification_report(all_targets, all_preds, output_dict=True)
        report_file = Path(tempfile.gettempdir()) / f"{model_name}_classification_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_file), artifact_path="reports")
        report_file.unlink()

        # Save confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        cm_file = Path(tempfile.gettempdir()) / f"{model_name}_confusion_matrix.npy"
        np.save(cm_file, cm)
        mlflow.log_artifact(str(cm_file), artifact_path="reports")
        cm_file.unlink()

        # Get class names (assuming EuroSAT dataset structure)
        class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]

        # Create visualization
        plot_path = create_prediction_visualization(
            test_dataset,
            all_preds,
            all_targets,
            all_probs,
            class_names,
            model_name
        )
        mlflow.log_artifact(plot_path, artifact_path="visualizations")
        Path(plot_path).unlink()

        # Get current run ID
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id if active_run else "unknown"

        return {
            "model_name": model_name,
            "config_run_id": config_run_id,
            "evaluation_run_id": run_id,
            "test_metrics": test_metrics,
            "hyperparameters": hyperparams
        }


def create_prediction_visualization(
    test_dataset: torch.utils.data.Dataset,
    predictions: List[int],
    targets: List[int],
    probabilities: List[np.ndarray],
    class_names: List[str],
    model_name: str,
    num_samples: int = 16
) -> str:
    """
    Create a beautiful visualization showing images, true labels, and predictions.

    Args:
        test_dataset: Test dataset
        predictions: Model predictions
        targets: True labels
        probabilities: Prediction probabilities
        class_names: List of class names
        model_name: Name of the model
        num_samples: Number of samples to visualize

    Returns:
        Path to the saved PNG file
    """
    # Select a diverse set of samples (some correct, some incorrect)
    correct_indices = [i for i, (p, t) in enumerate(zip(predictions, targets)) if p == t]
    incorrect_indices = [i for i, (p, t) in enumerate(zip(predictions, targets)) if p != t]

    # Try to get half correct and half incorrect
    num_correct = min(num_samples // 2, len(correct_indices))
    num_incorrect = min(num_samples - num_correct, len(incorrect_indices))

    # Randomly select samples
    np.random.seed(42)
    selected_correct = np.random.choice(correct_indices, num_correct, replace=False).tolist() if correct_indices else []
    selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False).tolist() if incorrect_indices else []
    selected_indices = selected_correct + selected_incorrect
    np.random.shuffle(selected_indices)

    # If we don't have enough samples, just use the first num_samples
    if len(selected_indices) < num_samples:
        try:
            dataset_len = len(test_dataset)  # type: ignore
        except TypeError:
            # If dataset doesn't support len, estimate from predictions
            dataset_len = len(predictions)
        selected_indices = list(range(min(num_samples, dataset_len)))

    # Create figure
    n_cols = 4
    n_rows = (len(selected_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'{model_name.upper()} - Test Set Predictions', fontsize=20, fontweight='bold')

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, sample_idx in enumerate(selected_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get image and denormalize
        # Handle different dataset structures
        if hasattr(test_dataset, 'dataset'):
            image, _ = test_dataset.dataset[sample_idx]  # type: ignore[attr-defined]
        else:
            image, _ = test_dataset[sample_idx]

        # Denormalize image (reverse the normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
        else:
            img_np = np.array(image)

        # Handle different shapes
        if img_np.shape[0] == 3:  # (C, H, W)
            img_np = img_np.transpose(1, 2, 0)  # (H, W, C)

        # Denormalize
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Display image
        ax.imshow(img_np)
        ax.axis('off')

        # Get prediction info
        true_label = targets[sample_idx]
        pred_label = predictions[sample_idx]
        prob = probabilities[sample_idx][pred_label]

        # Create title with color coding
        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'

        title = f"True: {class_names[true_label]}\n"
        title += f"Pred: {class_names[pred_label]} ({prob:.2%})"

        ax.set_title(title, fontsize=10, color=color, fontweight='bold')

    # Hide empty subplots
    for idx in range(len(selected_indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path(tempfile.gettempdir()) / f"{model_name}_predictions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved prediction visualization to {output_path}")

    return str(output_path)


def evaluate_mlp(
    best_models_df: pd.DataFrame,
    dev_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate MLP model on test dataset."""
    model_row = best_models_df[best_models_df["Model"] == "mlp"].iloc[0]
    config_run_id = model_row["Config_Run_ID"]

    return evaluate_best_model(
        "mlp",
        config_run_id,
        dev_dataset,
        test_dataset,
        params["models"]["mlp"],
        params["training"],
        params["data_loaders"],
        params["mlflow"]
    )


def evaluate_vgg16(
    best_models_df: pd.DataFrame,
    dev_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate VGG16 model on test dataset."""
    model_row = best_models_df[best_models_df["Model"] == "vgg16"].iloc[0]
    config_run_id = model_row["Config_Run_ID"]

    return evaluate_best_model(
        "vgg16",
        config_run_id,
        dev_dataset,
        test_dataset,
        params["models"]["vgg16"],
        params["training"],
        params["data_loaders"],
        params["mlflow"]
    )


def evaluate_resnet50(
    best_models_df: pd.DataFrame,
    dev_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate ResNet50 model on test dataset."""
    model_row = best_models_df[best_models_df["Model"] == "resnet50"].iloc[0]
    config_run_id = model_row["Config_Run_ID"]

    return evaluate_best_model(
        "resnet50",
        config_run_id,
        dev_dataset,
        test_dataset,
        params["models"]["resnet50"],
        params["training"],
        params["data_loaders"],
        params["mlflow"]
    )


def evaluate_vit(
    best_models_df: pd.DataFrame,
    dev_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate ViT model on test dataset."""
    model_row = best_models_df[best_models_df["Model"] == "vit_b_16"].iloc[0]
    config_run_id = model_row["Config_Run_ID"]

    return evaluate_best_model(
        "vit_b_16",
        config_run_id,
        dev_dataset,
        test_dataset,
        params["models"]["vit_b_16"],
        params["training"],
        params["data_loaders"],
        params["mlflow"]
    )
