"""
Fase 1: Hyperparameter Tuning usando Grid Search CV - AG_NEWS

Busca exaustiva de hiperparâmetros usando validação cruzada (3-fold CV)
com rastreamento completo via MLflow para o dataset AG_NEWS.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import mlflow
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)

from src.config import MLFLOW_TRACKING_URI, RESULTS_DIR
from src.hyperparameters_agnews import (
    HYPERPARAMETER_GRIDS_AGNEWS,
    CV_FOLDS_AGNEWS,
    SCORING_METRICS_AGNEWS,
    get_total_combinations_agnews,
)
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

# Diretório para salvar melhores parâmetros
BEST_PARAMS_DIR = RESULTS_DIR / "best_params_agnews"
BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_instance(model_name: str):
    """
    Retorna instância do modelo baseado no nome

    Args:
        model_name: Nome do modelo

    Returns:
        Instância do modelo
    """
    models = {
        "naive_bayes": NaiveBayesGaussian(),
        "naive_bayes_bernoulli": NaiveBayesBernoulli(),
        "naive_bayes_multinomial": NaiveBayesMultinomial(),
        "gmm": GMMClassifier(verbose=0),
        "logistic_softmax": LogisticRegressionSoftmax(verbose=0),
        "logistic_ovr": LogisticRegressionOvR(verbose=0),
        # Random Forest: n_jobs=1 to avoid nested parallelism with GridSearchCV
        # (GridSearchCV already uses n_jobs=-1 for parallel grid search)
        "random_forest": RandomForest(verbose=0, n_jobs=1),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name]


def run_grid_search_cv_agnews(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 3,
    n_jobs: int = -1,
    verbose: int = 2,
) -> Tuple[Dict[str, Any], float, str]:
    """
    Executa Grid Search CV para um modelo e loga resultados no MLflow

    Args:
        model_name: Nome do modelo
        X_train: Features de treino
        y_train: Labels de treino
        cv_folds: Número de folds para CV
        n_jobs: Número de jobs paralelos
        verbose: Nível de verbosidade

    Returns:
        Tuple (best_params, best_score, mlflow_run_id)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"GRID SEARCH CV - {model_name.upper()}")
    logger.info(f"{'=' * 80}")

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("hyperparameter-tuning-agnews")

    # Obter modelo e grid
    model = get_model_instance(model_name)
    param_grid = HYPERPARAMETER_GRIDS_AGNEWS[model_name]
    total_combinations = get_total_combinations_agnews(model_name)

    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Total combinations: {total_combinations}")
    logger.info(f"CV folds: {cv_folds}")
    logger.info(f"Total fits: {total_combinations * cv_folds}")

    # Configurar scoring
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }

    # Grid Search com Stratified K-Fold
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="accuracy",  # Refit usando accuracy
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    # Executar Grid Search
    logger.info("\nStarting Grid Search CV...")
    logger.info(f"Dataset shape: {X_train.shape}")
    logger.info(f"This will take approximately:")

    # Estimate time based on model complexity
    time_estimates = {
        "naive_bayes": "5-10 minutes",
        "naive_bayes_bernoulli": "5-10 minutes",
        "naive_bayes_multinomial": "5-10 minutes",
        "gmm": "15-30 minutes",
        "logistic_softmax": "10-20 minutes",
        "logistic_ovr": "10-20 minutes",
        "random_forest": "30-60 minutes (depending on subset size)",
    }
    logger.info(f"  Estimated time: {time_estimates.get(model_name, 'unknown')}")
    logger.info(f"  Progress will be shown below by GridSearchCV (verbose={verbose})")
    logger.info("")

    start_time = time.time()

    with mlflow.start_run(run_name=f"{model_name}_gridsearch") as parent_run:
        # Add MLflow tags for organization
        def get_model_family_local(name: str) -> str:
            if "naive_bayes" in name:
                return "naive_bayes"
            elif name == "gmm":
                return "gmm"
            elif "logistic" in name:
                return "logistic"
            elif "random_forest" in name:
                return "random_forest"
            return "unknown"

        def get_model_type_local(name: str) -> str:
            generative = ["naive_bayes", "naive_bayes_bernoulli", "naive_bayes_multinomial", "gmm"]
            return "generative" if name in generative else "discriminative"

        mlflow.set_tags({
            "experiment_type": "hyperparameter_tuning",
            "model": model_name,
            "model_family": get_model_family_local(model_name),
            "model_type": get_model_type_local(model_name),
            "dataset": "ag_news",
            "tuning_method": "grid_search",
            "cv_folds": str(cv_folds),
            "scoring_metric": "accuracy",
        })
        # Fit
        grid_search.fit(X_train, y_train)
        total_time = time.time() - start_time

        # Melhores resultados
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_index = grid_search.best_index_

        logger.info(f"\n{'=' * 80}")
        logger.info(f"GRID SEARCH COMPLETED - {model_name.upper()}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Best CV accuracy: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Log parent run (summary)
        mlflow.log_params(
            {
                "model_name": model_name,
                "cv_folds": cv_folds,
                "n_combinations": total_combinations,
                "total_fits": total_combinations * cv_folds,
            }
        )

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Log CV statistics (mean ± std) + coefficient of variation
        std_test_accuracy = grid_search.cv_results_["std_test_accuracy"][best_index]
        mlflow.log_metrics(
            {
                "best_cv_accuracy": best_score,
                "best_cv_accuracy_std": std_test_accuracy,
                "best_cv_precision_macro": grid_search.cv_results_["mean_test_precision_macro"][
                    best_index
                ],
                "best_cv_precision_macro_std": grid_search.cv_results_["std_test_precision_macro"][
                    best_index
                ],
                "best_cv_recall_macro": grid_search.cv_results_["mean_test_recall_macro"][
                    best_index
                ],
                "best_cv_recall_macro_std": grid_search.cv_results_["std_test_recall_macro"][
                    best_index
                ],
                "best_cv_f1_macro": grid_search.cv_results_["mean_test_f1_macro"][best_index],
                "best_cv_f1_macro_std": grid_search.cv_results_["std_test_f1_macro"][best_index],
                "total_time_seconds": total_time,
                "cv_coefficient_of_variation": (
                    std_test_accuracy / best_score if best_score > 0 else 0
                ),
            }
        )

        # Log best model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")

        # Salvar CV results como artifact
        cv_results_path = BEST_PARAMS_DIR / f"{model_name}_cv_results.json"
        with open(cv_results_path, "w") as f:
            # Converter numpy arrays para listas
            cv_results_clean = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in grid_search.cv_results_.items()
            }
            json.dump(cv_results_clean, f, indent=2)
        mlflow.log_artifact(str(cv_results_path))

        # Create CV score distribution visualization
        try:
            import matplotlib.pyplot as plt

            # Get all CV scores
            test_scores = grid_search.cv_results_["mean_test_accuracy"]
            test_stds = grid_search.cv_results_["std_test_accuracy"]

            plt.figure(figsize=(12, 6))
            plt.errorbar(
                range(len(test_scores)),
                test_scores,
                yerr=test_stds,
                fmt='o',
                capsize=5,
                alpha=0.7
            )
            plt.axhline(y=best_score, color='r', linestyle='--', label=f'Best: {best_score:.4f}')
            plt.xlabel('Configuration Index')
            plt.ylabel('CV Accuracy (mean ± std)')
            plt.title(f'CV Score Distribution - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            cv_dist_path = BEST_PARAMS_DIR / f"{model_name}_cv_distribution.png"
            plt.savefig(cv_dist_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(str(cv_dist_path))
            plt.close()

            logger.info(f"  Saved CV distribution plot to {cv_dist_path}")
        except Exception as e:
            logger.warning(f"Could not create CV distribution plot: {e}")

        # Log nested runs (top 5 configs)
        logger.info("\nTop 5 configurations:")
        top_indices = np.argsort(grid_search.cv_results_["mean_test_accuracy"])[::-1][:5]

        for rank, idx in enumerate(top_indices, 1):
            params = grid_search.cv_results_["params"][idx]
            mean_acc = grid_search.cv_results_["mean_test_accuracy"][idx]
            std_acc = grid_search.cv_results_["std_test_accuracy"][idx]

            logger.info(f"  {rank}. Accuracy: {mean_acc:.4f} (±{std_acc:.4f}) | {params}")

            with mlflow.start_run(
                run_name=f"{model_name}_rank{rank}", nested=True
            ) as nested_run:
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        "cv_accuracy_mean": mean_acc,
                        "cv_accuracy_std": std_acc,
                        "cv_precision_macro_mean": grid_search.cv_results_[
                            "mean_test_precision_macro"
                        ][idx],
                        "cv_recall_macro_mean": grid_search.cv_results_[
                            "mean_test_recall_macro"
                        ][idx],
                        "cv_f1_macro_mean": grid_search.cv_results_["mean_test_f1_macro"][idx],
                        "rank": rank,
                    }
                )

        # Salvar best params em JSON
        best_params_path = BEST_PARAMS_DIR / f"{model_name}_best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        mlflow.log_artifact(str(best_params_path))

        logger.info(f"\nBest parameters saved to: {best_params_path}")
        logger.info(f"MLflow run ID: {parent_run.info.run_id}")

        return best_params, best_score, parent_run.info.run_id


def tune_all_models_agnews(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: list = None,
    cv_folds: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Executa Grid Search CV para todos os modelos

    Args:
        X_train: Features de treino
        y_train: Labels de treino
        models: Lista de modelos (None = todos)
        cv_folds: Número de folds

    Returns:
        Dicionário com melhores parâmetros por modelo
    """
    if models is None:
        models = [
            "naive_bayes",
            "naive_bayes_bernoulli",
            "naive_bayes_multinomial",
            "gmm",
            "logistic_softmax",
            "logistic_ovr",
            "random_forest",
        ]

    all_best_params = {}

    for model_name in models:
        try:
            best_params, best_score, run_id = run_grid_search_cv_agnews(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                cv_folds=cv_folds,
            )

            all_best_params[model_name] = {
                "params": best_params,
                "cv_score": best_score,
                "run_id": run_id,
            }

        except Exception as e:
            logger.error(f"Error tuning {model_name}: {e}")
            raise

    # Salvar todos os best params juntos
    all_params_path = BEST_PARAMS_DIR / "all_best_params.json"
    with open(all_params_path, "w") as f:
        # Serializar apenas params e score (run_id é string)
        serializable = {
            k: {"params": v["params"], "cv_score": float(v["cv_score"])}
            for k, v in all_best_params.items()
        }
        json.dump(serializable, f, indent=2)

    logger.info(f"\n{'=' * 80}")
    logger.info("ALL MODELS TUNED")
    logger.info(f"{'=' * 80}")
    logger.info(f"Best parameters saved to: {all_params_path}")

    return all_best_params


if __name__ == "__main__":
    # Teste com subset pequeno
    from src.data_loader import FashionMNISTLoader

    logger.info("Loading Fashion MNIST...")
    loader = FashionMNISTLoader(flatten=True, normalize=True)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    # Usar subset para teste rápido
    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]

    logger.info(f"Train shape: {X_train_small.shape}")

    # Testar um modelo
    best_params, best_score, run_id = run_grid_search_cv_agnews(
        model_name="naive_bayes",
        X_train=X_train_small,
        y_train=y_train_small,
        cv_folds=3,  # Menos folds para teste rápido
        verbose=1,
    )

    logger.info("\nTest completed successfully!")
