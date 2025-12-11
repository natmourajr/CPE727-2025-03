"""
Fase 2: Avaliação Final no Test Set

Carrega os melhores hiperparâmetros da Fase 1, retreina os modelos
com train+val completo, e avalia no test set.

Usa MLflow Artifacts (Opção 3) para carregar melhores modelos.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import MLFLOW_TRACKING_URI, RESULTS_DIR
from src.models import (
    NaiveBayesGaussian,
    NaiveBayesBernoulli,
    NaiveBayesMultinomial,
    GMMClassifier,
    LogisticRegressionSoftmax,
    LogisticRegressionOvR,
    RandomForest,
)
from src.utils import setup_logger

logger = setup_logger(__name__)

BEST_PARAMS_DIR = RESULTS_DIR / "best_params"


def load_best_params(model_name: str) -> Dict[str, Any]:
    """
    Carrega melhores hiperparâmetros da Fase 1

    Args:
        model_name: Nome do modelo

    Returns:
        Dicionário com melhores parâmetros
    """
    params_file = BEST_PARAMS_DIR / f"{model_name}_best_params.json"

    if not params_file.exists():
        raise FileNotFoundError(
            f"Best parameters not found for {model_name}. "
            f"Run hyperparameter tuning first (Phase 1)."
        )

    with open(params_file, "r") as f:
        params = json.load(f)

    logger.info(f"Loaded best params for {model_name}: {params}")
    return params


def load_model_from_run(run_id: str, artifact_path: str = "best_model"):
    """
    Carrega modelo do MLflow usando run_id (Opção 3)

    Args:
        run_id: ID do run MLflow
        artifact_path: Caminho do artifact

    Returns:
        Modelo carregado
    """
    model_uri = f"runs:/{run_id}/best_model"
    logger.info(f"Loading model from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        raise


def get_model_instance(model_name: str, params: Dict[str, Any]):
    """
    Cria instância do modelo com parâmetros fornecidos

    Args:
        model_name: Nome do modelo
        params: Hiperparâmetros

    Returns:
        Instância do modelo
    """
    models_map = {
        # Generativos
        "naive_bayes": NaiveBayesGaussian,
        "naive_bayes_bernoulli": NaiveBayesBernoulli,
        "naive_bayes_multinomial": NaiveBayesMultinomial,
        "gmm": GMMClassifier,
        # Discriminativos
        "logistic_softmax": LogisticRegressionSoftmax,
        "logistic_ovr": LogisticRegressionOvR,
        "random_forest": RandomForest,
    }

    if model_name not in models_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_map.keys())}")

    # Adicionar parâmetros fixos
    if model_name in ["gmm", "logistic_softmax", "logistic_ovr"]:
        params["verbose"] = 0
        params["random_state"] = 42
    elif model_name == "random_forest":
        params["random_state"] = 42
        params["n_jobs"] = -1

    return models_map[model_name](**params)


def evaluate_final_model(
    model_name: str,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Optional[Dict[str, Any]] = None,
    run_id_from_phase1: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Avalia modelo final no test set usando melhores hiperparâmetros

    Args:
        model_name: Nome do modelo
        X_train_full: Train + Val combinados (60k amostras)
        y_train_full: Labels train + val
        X_test: Test set
        y_test: Labels test
        best_params: Melhores parâmetros (se None, carrega do arquivo)
        run_id_from_phase1: Run ID da Fase 1 (opcional)

    Returns:
        Dicionário com métricas e resultados
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FINAL EVALUATION - {model_name.upper()}")
    logger.info(f"{'=' * 80}")

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("final-evaluation")

    # Carregar melhores parâmetros se não fornecidos
    if best_params is None:
        best_params = load_best_params(model_name)

    logger.info(f"Using parameters: {best_params}")
    logger.info(f"Training on full dataset: {X_train_full.shape}")
    logger.info(f"Testing on: {X_test.shape}")

    with mlflow.start_run(run_name=f"{model_name}_final") as run:
        # Criar modelo com melhores parâmetros
        model = get_model_instance(model_name, best_params)

        # Log params
        mlflow.log_params({"model_name": model_name, **best_params})
        mlflow.log_params(
            {
                "train_size": len(X_train_full),
                "test_size": len(X_test),
                "n_features": X_train_full.shape[1],
            }
        )

        # Link para run da Fase 1 (se fornecido)
        if run_id_from_phase1:
            mlflow.set_tag("phase1_run_id", run_id_from_phase1)

        # Treinar
        logger.info("\nTraining final model on train+val...")
        start_time = time.time()
        model.fit(X_train_full, y_train_full)
        train_time = time.time() - start_time

        logger.info(f"Training completed in {train_time:.2f}s")

        # Predições no test set
        logger.info("\nEvaluating on test set...")
        start_time = time.time()
        y_test_pred = model.predict(X_test)
        inference_time_total = time.time() - start_time
        inference_time_per_sample = inference_time_total / len(X_test)

        # Probabilidades (se disponível)
        try:
            y_test_proba = model.predict_proba(X_test)
        except AttributeError:
            y_test_proba = None

        # Métricas
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision_macro = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        test_recall_macro = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
        test_f1_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        test_precision_weighted = precision_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        test_recall_weighted = recall_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        )
        test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

        # Log métricas
        metrics = {
            "test_accuracy": test_accuracy,
            "test_precision_macro": test_precision_macro,
            "test_recall_macro": test_recall_macro,
            "test_f1_macro": test_f1_macro,
            "test_precision_weighted": test_precision_weighted,
            "test_recall_weighted": test_recall_weighted,
            "test_f1_weighted": test_f1_weighted,
            "train_time_seconds": train_time,
            "inference_time_seconds": inference_time_total,
            "inference_time_per_sample_ms": inference_time_per_sample * 1000,
        }

        mlflow.log_metrics(metrics)

        # Log modelo final
        mlflow.sklearn.log_model(model, "final_model")

        # Salvar confusion matrix
        from src.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker()
        tracker._log_confusion_matrix(y_test, y_test_pred, f"{model_name}_final")
        tracker._log_classification_report(y_test, y_test_pred, f"{model_name}_final")

        # Resultados
        logger.info(f"\n{'=' * 80}")
        logger.info("FINAL TEST RESULTS")
        logger.info(f"{'=' * 80}")
        logger.info(f"Accuracy: {test_accuracy:.4f}")
        logger.info(f"Precision (macro): {test_precision_macro:.4f}")
        logger.info(f"Recall (macro): {test_recall_macro:.4f}")
        logger.info(f"F1-score (macro): {test_f1_macro:.4f}")
        logger.info(f"Training time: {train_time:.2f}s")
        logger.info(f"Inference time: {inference_time_per_sample * 1000:.2f}ms/sample")
        logger.info(f"\nMLflow run ID: {run.info.run_id}")

        results = {
            "model_name": model_name,
            "best_params": best_params,
            "metrics": metrics,
            "run_id": run.info.run_id,
            "predictions": y_test_pred,
            "probabilities": y_test_proba,
        }

        return results


def evaluate_all_models(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: list = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Avalia todos os modelos no test set

    Args:
        X_train_full: Train + val combinados
        y_train_full: Labels train + val
        X_test: Test set
        y_test: Labels test
        models: Lista de modelos (None = todos)

    Returns:
        Dicionário com resultados por modelo
    """
    if models is None:
        models = ["naive_bayes", "gmm", "logistic_softmax", "logistic_ovr"]

    all_results = {}

    for model_name in models:
        try:
            results = evaluate_final_model(
                model_name=model_name,
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
            )

            all_results[model_name] = results

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise

    # Comparação final
    logger.info(f"\n{'=' * 80}")
    logger.info("FINAL COMPARISON - ALL MODELS")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Model':<20} {'Accuracy':<12} {'F1 (macro)':<12} {'Train Time':<12}")
    logger.info(f"{'-' * 80}")

    for model_name, results in all_results.items():
        metrics = results["metrics"]
        logger.info(
            f"{model_name:<20} "
            f"{metrics['test_accuracy']:<12.4f} "
            f"{metrics['test_f1_macro']:<12.4f} "
            f"{metrics['train_time_seconds']:<12.2f}s"
        )

    # Salvar resultados
    results_path = RESULTS_DIR / "final_test_results.json"
    with open(results_path, "w") as f:
        # Remover arrays numpy para serialização
        serializable = {
            k: {
                "model_name": v["model_name"],
                "best_params": v["best_params"],
                "metrics": v["metrics"],
                "run_id": v["run_id"],
            }
            for k, v in all_results.items()
        }
        json.dump(serializable, f, indent=2)

    logger.info(f"\nFinal results saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    # Teste
    from src.data_loader import FashionMNISTLoader

    logger.info("Loading Fashion MNIST...")
    loader = FashionMNISTLoader(flatten=True, normalize=True)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    # Combinar train + val
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])

    logger.info(f"Train+Val: {X_train_full.shape}")
    logger.info(f"Test: {X_test.shape}")

    # Testar com Naive Bayes (precisa ter rodado Phase 1 primeiro)
    # results = evaluate_final_model(
    #     model_name="naive_bayes",
    #     X_train_full=X_train_full,
    #     y_train_full=y_train_full,
    #     X_test=X_test,
    #     y_test=y_test,
    # )

    logger.info("\nFinal evaluation module ready!")
    logger.info("Run hyperparameter_tuning.py first (Phase 1), then use this module (Phase 2).")
