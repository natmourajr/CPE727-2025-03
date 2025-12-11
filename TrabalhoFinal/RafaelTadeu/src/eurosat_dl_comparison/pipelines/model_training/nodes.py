"""Training nodes with MLFlow integration."""
import json
import logging
import tempfile
from pathlib import Path
from itertools import product
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

from .models import create_model

logger = logging.getLogger(__name__)


def train_model_with_gridsearch(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    model_params: Dict[str, Any],
    training_params: Dict[str, Any],
    mlflow_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Train model with gridsearch and MLFlow logging."""

    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    gridsearch_params = model_params["gridsearch"]
    param_names = list(gridsearch_params.keys())
    param_values = [gridsearch_params[name] for name in param_names]
    param_combinations = list(product(*param_values))

    logger.info(f"Starting gridsearch for {model_name} with {len(param_combinations)} combinations")

    best_model_path = None
    best_accuracy = 0.0

    with mlflow.start_run(run_name=f"{model_name}_gridsearch"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("total_combinations", len(param_combinations))

        for idx, param_combo in enumerate(param_combinations):
            params_dict = dict(zip(param_names, param_combo))

            with mlflow.start_run(run_name=f"config_{idx}", nested=True):
                logger.info(f"Training {model_name} with params: {params_dict}")

                for key, value in params_dict.items():
                    mlflow.log_param(key, value)

                model_config = {**model_params, **params_dict}
                model_config.pop("gridsearch", None)

                model = create_model(model_name, model_config)
                device = torch.device(training_params["device"] if torch.cuda.is_available() else "cpu")
                model = model.to(device)

                learning_rate = params_dict.get("learning_rate", 0.001)
                batch_size = params_dict.get("batch_size", 32)

                mlflow.log_param("epochs", training_params["epochs"])
                mlflow.log_param("weight_decay", training_params["weight_decay"])

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=training_params["weight_decay"]
                )
                criterion = nn.CrossEntropyLoss()

                best_val_loss = float('inf')
                patience_counter = 0

                for epoch in range(training_params["epochs"]):
                    model.train()
                    train_loss = 0.0

                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()

                    avg_train_loss = train_loss / len(train_loader)

                    model.eval()
                    val_loss = 0.0
                    all_preds = []
                    all_targets = []

                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output, target)
                            val_loss += loss.item()
                            preds = output.argmax(dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(target.cpu().numpy())

                    avg_val_loss = val_loss / len(val_loader)
                    val_accuracy = accuracy_score(all_targets, all_preds)

                    if epoch % 10 == 0:
                        logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_accuracy:.4f}")

                    if avg_val_loss < best_val_loss - training_params["early_stopping"]["min_delta"]:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= training_params["early_stopping"]["patience"]:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break

                precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

                metrics = {
                    "final_train_loss": avg_train_loss,
                    "final_val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1": f1
                }

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(metrics, f, indent=2)
                    mlflow.log_artifact(f.name, "metrics")
                    Path(f.name).unlink()

                report = classification_report(all_targets, all_preds, output_dict=True)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(report, f, indent=2)
                    mlflow.log_artifact(f.name, "classification_report")
                    Path(f.name).unlink()

                with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                    torch.save(model.state_dict(), f.name)
                    mlflow.log_artifact(f.name, "model")

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_model_path = f.name
                    else:
                        Path(f.name).unlink()

        logger.info(f"Best accuracy for {model_name}: {best_accuracy:.4f}")

    return {
        "model_name": model_name,
        "best_accuracy": best_accuracy,
        "best_model_path": best_model_path
    }


class TransformSubset(torch.utils.data.Dataset):
    """Wrapper to apply transforms to a subset of a dataset."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.subset.dataset[idx]
        # Apply our transform
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def train_model_with_cv(
    dev_dataset: torch.utils.data.Dataset,
    cross_val_table: pd.DataFrame,
    model_name: str,
    model_params: Dict[str, Any],
    training_params: Dict[str, Any],
    data_loader_params: Dict[str, Any],
    mlflow_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Train model with cross-validation and gridsearch."""
    from torch.utils.data import Subset
    from torchvision import transforms

    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    gridsearch_params = model_params["gridsearch"]
    param_names = list(gridsearch_params.keys())
    param_values = [gridsearch_params[name] for name in param_names]
    param_combinations = list(product(*param_values))

    n_folds = training_params["cv_folds"]
    logger.info(f"Starting CV for {model_name} with {len(param_combinations)} hyperparameter combinations and {n_folds} folds")

    # Transforms for tensors (data already converted to tensor during download)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            normalize,
        ]
    )

    best_config = None
    best_avg_accuracy = 0.0

    with mlflow.start_run(run_name=f"{model_name}_cv_gridsearch"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("total_combinations", len(param_combinations))
        mlflow.log_param("cv_folds", n_folds)

        config_pbar = tqdm(enumerate(param_combinations), total=len(param_combinations),
                          desc=f"{model_name} Gridsearch", unit="config")

        for idx, param_combo in config_pbar:
            params_dict = dict(zip(param_names, param_combo))

            with mlflow.start_run(run_name=f"config_{idx}", nested=True):
                logger.info(f"Evaluating {model_name} config {idx+1}/{len(param_combinations)}: {params_dict}")

                for key, value in params_dict.items():
                    mlflow.log_param(key, value)

                fold_metrics = []

                fold_pbar = tqdm(range(n_folds), desc=f"Config {idx+1} Folds", leave=False, unit="fold")
                for fold in fold_pbar:
                    with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                        mlflow.log_param("fold", fold)

                        train_indices = cross_val_table[cross_val_table["fold"] != fold]["sample_idx"].tolist()
                        val_indices = cross_val_table[cross_val_table["fold"] == fold]["sample_idx"].tolist()

                        train_subset = Subset(dev_dataset, train_indices)
                        val_subset = Subset(dev_dataset, val_indices)

                        # Create subsets with different transforms
                        train_subset = TransformSubset(
                            train_subset, train_transform
                        )
                        val_subset = TransformSubset(
                            val_subset, eval_transform
                        )

                        batch_size = params_dict.get("batch_size", data_loader_params["batch_size"])
                        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                                 num_workers=data_loader_params["num_workers"],
                                                 pin_memory=data_loader_params["pin_memory"])
                        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                               num_workers=data_loader_params["num_workers"],
                                               pin_memory=data_loader_params["pin_memory"])

                        print(f"Image shape: {train_subset[0][0].shape}")

                        model_config = {**model_params, **params_dict}
                        model_config.pop("gridsearch", None)
                        model = create_model(model_name, model_config)
                        device = torch.device(training_params["device"] if torch.cuda.is_available() else "cpu")
                        model = model.to(device)

                        learning_rate = params_dict.get("learning_rate", 0.001)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                                    weight_decay=training_params["weight_decay"])
                        criterion = nn.CrossEntropyLoss()

                        best_val_loss = float('inf')
                        best_train_loss = float('inf')
                        best_model_state = None
                        patience_counter = 0

                        epoch_pbar = tqdm(range(training_params["epochs"]),
                                        desc=f"Fold {fold} Training", leave=False, unit="epoch")

                        # Convert history to metric-based format
                        history_dict = {
                            "train_loss": [],
                            "val_loss": [],
                            "val_accuracy": [],
                            "val_precision": [],
                            "val_recall": [],
                            "val_f1": []
                        }
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

                            model.eval()
                            val_loss = 0.0
                            all_preds = []
                            all_targets = []
                            with torch.no_grad():
                                val_batch_pbar = tqdm(val_loader, desc="Val", leave=False, unit="batch")
                                for data, target in val_batch_pbar:
                                    data, target = data.to(device), target.to(device)
                                    output = model(data)
                                    loss = criterion(output, target)
                                    val_loss += loss.item()
                                    preds = output.argmax(dim=1)
                                    all_preds.extend(preds.cpu().numpy())
                                    all_targets.extend(target.cpu().numpy())
                                    val_batch_pbar.set_postfix(loss=loss.item())

                            avg_val_loss = val_loss / len(val_loader)
                            val_accuracy = accuracy_score(all_targets, all_preds)
                            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

                            # Record epoch history
                            history_dict["train_loss"].append(avg_train_loss)
                            history_dict["val_loss"].append(avg_val_loss)
                            history_dict["val_accuracy"].append(val_accuracy)
                            history_dict["val_precision"].append(precision)
                            history_dict["val_recall"].append(recall)
                            history_dict["val_f1"].append(f1)

                            epoch_pbar.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss, val_acc=val_accuracy)

                            if avg_val_loss < best_val_loss - training_params["early_stopping"]["min_delta"]:
                                best_val_loss = avg_val_loss
                                best_train_loss = avg_train_loss
                                best_model_state = model.state_dict().copy()
                                patience_counter = 0
                            else:
                                patience_counter += 1
                                if patience_counter >= training_params["early_stopping"]["patience"]:
                                    logger.info(f"Fold {fold} early stopping at epoch {epoch}")
                                    break

                        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

                        metrics = {
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                            "best_train_loss": best_train_loss,
                            "best_val_loss": best_val_loss,
                            "val_accuracy": val_accuracy,
                            "val_precision": precision,
                            "val_recall": recall,
                            "val_f1": f1
                        }
                        fold_metrics.append(metrics)
                        fold_pbar.set_postfix(acc=val_accuracy, f1=f1)

                        # Log metrics to MLFlow
                        mlflow.log_metric("train_loss", avg_train_loss)
                        mlflow.log_metric("val_loss", avg_val_loss)
                        mlflow.log_metric("best_train_loss", best_train_loss)
                        mlflow.log_metric("best_val_loss", best_val_loss)
                        mlflow.log_metric("val_accuracy", val_accuracy)
                        mlflow.log_metric("val_precision", precision)
                        mlflow.log_metric("val_recall", recall)
                        mlflow.log_metric("val_f1", f1)

                        # Log complete training history as artifact
                        history_filename = f"fold_{fold}_history.json"
                        temp_history_path = Path(tempfile.gettempdir()) / history_filename
                        with open(temp_history_path, 'w') as f:
                            json.dump(history_dict, f, indent=2)
                        mlflow.log_artifact(str(temp_history_path), artifact_path="training_history")
                        temp_history_path.unlink()

                        # Save best model state (not final model state)
                        if best_model_state is not None:
                            model_filename = f"fold_{fold}_best_model.pth"
                            temp_model_path = Path(tempfile.gettempdir()) / model_filename
                            torch.save(best_model_state, temp_model_path)
                            mlflow.log_artifact(str(temp_model_path), artifact_path="models")
                            temp_model_path.unlink()

                avg_metrics = {
                    "avg_val_accuracy": np.mean([m["val_accuracy"] for m in fold_metrics]),
                    "std_val_accuracy": np.std([m["val_accuracy"] for m in fold_metrics]),
                    "avg_val_f1": np.mean([m["val_f1"] for m in fold_metrics]),
                    "avg_val_precision": np.mean([m["val_precision"] for m in fold_metrics]),
                    "avg_val_recall": np.mean([m["val_recall"] for m in fold_metrics]),
                    "avg_best_train_loss": np.mean([m["best_train_loss"] for m in fold_metrics]),
                    "avg_best_val_loss": np.mean([m["best_val_loss"] for m in fold_metrics]),
                    "std_best_val_loss": np.std([m["best_val_loss"] for m in fold_metrics])
                }

                # Log aggregated metrics to MLFlow
                mlflow.log_metric("avg_best_train_loss", avg_metrics["avg_best_train_loss"])
                mlflow.log_metric("avg_best_val_loss", avg_metrics["avg_best_val_loss"])
                mlflow.log_metric("std_best_val_loss", avg_metrics["std_best_val_loss"])
                mlflow.log_metric("avg_val_accuracy", avg_metrics["avg_val_accuracy"])
                mlflow.log_metric("std_val_accuracy", avg_metrics["std_val_accuracy"])
                mlflow.log_metric("avg_val_f1", avg_metrics["avg_val_f1"])
                mlflow.log_metric("avg_val_precision", avg_metrics["avg_val_precision"])
                mlflow.log_metric("avg_val_recall", avg_metrics["avg_val_recall"])

                cv_summary_filename = f"config_{idx}_cv_summary.json"
                temp_cv_metrics_path = Path(tempfile.gettempdir()) / cv_summary_filename
                with open(temp_cv_metrics_path, 'w') as f:
                    json.dump(avg_metrics, f, indent=2)
                mlflow.log_artifact(str(temp_cv_metrics_path), artifact_path="cv_summary")
                temp_cv_metrics_path.unlink()

                logger.info(f"Config {idx}: avg_accuracy={avg_metrics['avg_val_accuracy']:.4f} ± {avg_metrics['std_val_accuracy']:.4f}, "
                           f"avg_best_val_loss={avg_metrics['avg_best_val_loss']:.4f}")
                config_pbar.set_postfix(best_acc=best_avg_accuracy, curr_acc=avg_metrics['avg_val_accuracy'])

                if avg_metrics["avg_val_accuracy"] > best_avg_accuracy:
                    best_avg_accuracy = avg_metrics["avg_val_accuracy"]
                    best_config = {"params": params_dict, "metrics": avg_metrics}

        logger.info(f"Best {model_name} config: accuracy={best_avg_accuracy:.4f}, params={best_config['params']}")

    return {"model_name": model_name, "best_config": best_config}


def train_mlp_cv(
    dev_dataset: torch.utils.data.Dataset,
    cross_val_table: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Train MLP with cross-validation."""
    return train_model_with_cv(
        dev_dataset, cross_val_table, "mlp",
        params["models"]["mlp"], params["training"],
        params["data_loaders"], params["mlflow"]
    )


def train_vgg16_cv(
    dev_dataset: torch.utils.data.Dataset,
    cross_val_table: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Train VGG16 with cross-validation."""
    return train_model_with_cv(
        dev_dataset, cross_val_table, "vgg16",
        params["models"]["vgg16"], params["training"],
        params["data_loaders"], params["mlflow"]
    )


def train_resnet50_cv(
    dev_dataset: torch.utils.data.Dataset,
    cross_val_table: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Train ResNet50 with cross-validation."""
    return train_model_with_cv(
        dev_dataset, cross_val_table, "resnet50",
        params["models"]["resnet50"], params["training"],
        params["data_loaders"], params["mlflow"]
    )


def train_vit_cv(
    dev_dataset: torch.utils.data.Dataset,
    cross_val_table: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Train ViT with cross-validation."""
    return train_model_with_cv(
        dev_dataset, cross_val_table, "vit_b_16",
        params["models"]["vit_b_16"], params["training"],
        params["data_loaders"], params["mlflow"]
    )
