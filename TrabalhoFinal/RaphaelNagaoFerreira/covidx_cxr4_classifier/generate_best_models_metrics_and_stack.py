import sys
import os
import json
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src/dataloaders/CovidxCxr4Loader'))
from covidx_cxr4_loader.loader import build_loaders

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src/models'))
from DenseNet.densenet import DenseNet121
from ViT.vit import ViTBinaryClassifier

# Import logger
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src/modules'))
from logger import ExperimentLogger

from train import MODELS_NAME, MODELS_TEST_TRANSFORMS
from train_funcs import find_best_threshold_youden, evaluate_test, plot_roc


def generate_best_models_metrics_and_stack(batch_size=16, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_root = os.path.join(os.path.dirname(__file__), '../../..')
    saved_data_path = os.path.join(repo_root, "TrabalhoFinal", "RaphaelNagaoFerreira", "covidx_cxr4_classifier", "saved_data")
    os.makedirs(saved_data_path, exist_ok=True)
    logger = ExperimentLogger('covidx_cxr4_bootstrap', results_dir=saved_data_path)

    model_names = ["densenet121", "vit", "unet"]

    models_stack = {}
    thresholds = {}
    models_metrics = {}
    models_probs = {}

    for model_name in model_names:
        logger.log(f"RUNNING {model_name}")
        model = MODELS_NAME[model_name](num_classes=1, device=device)
        model = load_best_model(model_name, model, saved_data_path, device, logger)
        if model is None:
            continue
        models_stack[model_name] = model

        test_transform = MODELS_TEST_TRANSFORMS[model_name]

        # --------------------------------------------------
        # Loaders (train / val / test)
        # --------------------------------------------------
        train_loader, val_loader, test_loader, pos_weight = build_loaders(
            batch_size=batch_size,
            num_workers=4,
            test_transforms=test_transform
        )

        threshold, score = find_best_threshold_youden(model, val_loader, device)
        thresholds[model_name] = threshold
        logger.log(f"Threshold {model_name}: {threshold}")

        best_model_metrics_path = os.path.join(saved_data_path, model_name, "best_model_bootstrap_metrics")
        os.makedirs(best_model_metrics_path, exist_ok=True)
        logger.log(f"Run bootstrap metrics for {model_name}")
        metrics, all_probs, all_labels = bootstrap_metric(model, test_loader, threshold, device, best_model_metrics_path, logger, n_boot=1)
        models_metrics[model_name] = metrics
        models_probs[model_name] = all_probs

    if models_probs == {}:
        return None

    stack_probs = np.mean(
        np.column_stack(list(models_probs.values())),
        axis=1
    )

    stack_dir = os.path.join(saved_data_path, "stack_model")
    os.makedirs(stack_dir, exist_ok=True)

    plot_roc(all_labels, stack_probs, stack_dir)
    stack_roc_preds = (stack_probs >= 0.5).astype(int)
    stack_metrics = {
        "ROC-AUC": roc_auc_score(all_labels, stack_probs),
        "Accuracy": accuracy_score(all_labels, stack_roc_preds),
        "Recall": recall_score(all_labels, stack_roc_preds),
        "Precision": precision_score(all_labels, stack_roc_preds),
        "F1": f1_score(all_labels, stack_roc_preds),
    }
    with open(os.path.join(stack_dir, 'stack_evaluation.json'), "w") as f:
        json.dump(stack_metrics, f)
    logger.log(f"FINISHED")


def load_best_model(model_name, model, saved_data_path, device, logger):
    best_model = os.path.join(saved_data_path, model_name, "best_model.pth")
    if not os.path.exists(best_model):
        logger.log(f"No best model found for {model_name}")
        logger.log(f"Generate a best_model.pth and save it in {best_model}")
        return None

    model.load_state_dict(torch.load(best_model, map_location=torch.device(device)))
    return model

def bootstrap_metric(model, test_loader, threshold, device, path, logger, n_boot=1000):
    metrics_scores = {}

    for _ in tqdm(range(n_boot)):
        metrics, all_probs, all_labels = evaluate_test(
            model=model,
            test_loader=test_loader,
            device=device,
            threshold=threshold,
            logger=logger,
            path_dir=path,
            max_saliency_images=0
        )
        for key, value in metrics.items():
            if key not in metrics_scores.keys():
                metrics_scores[key] = []
            metrics_scores[key].append(value)

    bootstrap_metrics = {}
    for key, value in metrics_scores.items():
        scores = np.array(value)
        bootstrap_metrics[key] = {"mean": scores.mean(), "std": scores.std()}

    with open(os.path.join(path, 'bootstrap_evaluation.json'), "w") as f:
        json.dump(bootstrap_metrics, f)

    return bootstrap_metrics, torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


if __name__ == '__main__':
    generate_best_models_metrics_and_stack()