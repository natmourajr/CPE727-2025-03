"""
Script Principal de Treinamento para AG_NEWS

Integra Fase 1 (Grid Search CV) e Fase 2 (Final Evaluation) para o dataset AG_NEWS

Uso:
    # Fase 1: Hyperparameter tuning
    uv run python src/train_agnews.py --phase 1 --model naive_bayes
    uv run python src/train_agnews.py --phase 1 --all

    # Fase 2: Final evaluation no test set
    uv run python src/train_agnews.py --phase 2 --model naive_bayes
    uv run python src/train_agnews.py --phase 2 --all

    # Ambas as fases
    uv run python src/train_agnews.py --full --model naive_bayes
    uv run python src/train_agnews.py --full --all
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.data_loader_agnews import AGNewsLoader
from src.hyperparameter_tuning_agnews import run_grid_search_cv_agnews, tune_all_models_agnews
from src.final_evaluation_agnews import evaluate_final_model_agnews, evaluate_all_models_agnews
from src.config import AVAILABLE_MODELS
from src.utils import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train and evaluate AG_NEWS classification models"
    )

    # Fase
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        help="Phase: 1=Hyperparameter tuning (CV), 2=Final evaluation (test set)",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run both phases (1 and 2) sequentially",
    )

    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        help="Specific model to train",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models",
    )

    # Parâmetros CV
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3 for AG_NEWS)",
    )

    # Subset para teste rápido
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use small subset for quick testing",
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        default=10000,
        help="Size of subset for quick test (default: 10000)",
    )

    # TF-IDF parameters
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Max TF-IDF features (default: 10000)",
    )

    args = parser.parse_args()

    # Validação
    if not args.phase and not args.full:
        parser.error("Must specify --phase or --full")

    if not args.model and not args.all:
        parser.error("Must specify --model or --all")

    if args.model and args.all:
        parser.error("Cannot specify both --model and --all")

    return args


def main():
    """Main training pipeline"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("AG_NEWS TRAINING PIPELINE")
    logger.info("=" * 80)

    # Carregar dados
    logger.info(f"\nLoading AG_NEWS dataset (max_features={args.max_features})...")
    loader = AGNewsLoader(max_features=args.max_features)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Val set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")

    # Quick test mode
    if args.quick_test:
        logger.info(f"\n⚡ QUICK TEST MODE - Using subset of {args.subset_size} samples")
        X_train = X_train[: args.subset_size]
        y_train = y_train[: args.subset_size]
        X_val = X_val[: args.subset_size // 4]
        y_val = y_val[: args.subset_size // 4]
        X_test = X_test[: args.subset_size // 5]
        y_test = y_test[: args.subset_size // 5]

    # Determinar modelos
    if args.all:
        models_to_train = AVAILABLE_MODELS
    else:
        models_to_train = [args.model]

    logger.info(f"\nModels to train: {models_to_train}")

    # Determinar fases
    if args.full:
        phases = [1, 2]
    else:
        phases = [args.phase]

    logger.info(f"Phases to run: {phases}")

    # FASE 1: Hyperparameter Tuning
    if 1 in phases:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: HYPERPARAMETER TUNING (Grid Search CV)")
        logger.info("=" * 80)

        if args.all:
            # Tunar todos os modelos
            all_best_params = tune_all_models_agnews(
                X_train=X_train,
                y_train=y_train,
                models=models_to_train,
                cv_folds=args.cv_folds,
            )

            logger.info("\n✓ Phase 1 completed for all models")

        else:
            # Tunar um modelo específico
            best_params, best_score, run_id = run_grid_search_cv_agnews(
                model_name=args.model,
                X_train=X_train,
                y_train=y_train,
                cv_folds=args.cv_folds,
            )

            logger.info(f"\n✓ Phase 1 completed for {args.model}")
            logger.info(f"  Best CV accuracy: {best_score:.4f}")
            logger.info(f"  Best params: {best_params}")

    # FASE 2: Final Evaluation
    if 2 in phases:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: FINAL EVALUATION (Test Set)")
        logger.info("=" * 80)

        # Combinar train + val
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])

        logger.info(f"Training on full dataset: {X_train_full.shape}")
        logger.info(f"Testing on: {X_test.shape}")

        if args.all:
            # Avaliar todos os modelos
            all_results = evaluate_all_models_agnews(
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
                models=models_to_train,
            )

            logger.info("\n✓ Phase 2 completed for all models")

            # Mostrar ranking
            logger.info("\n" + "=" * 80)
            logger.info("FINAL RANKING (by test accuracy)")
            logger.info("=" * 80)

            ranking = sorted(
                all_results.items(),
                key=lambda x: x[1]["metrics"]["test_accuracy"],
                reverse=True,
            )

            for rank, (model_name, results) in enumerate(ranking, 1):
                acc = results["metrics"]["test_accuracy"]
                f1 = results["metrics"]["test_f1_macro"]
                logger.info(f"{rank}. {model_name:<20} Acc: {acc:.4f}  F1: {f1:.4f}")

        else:
            # Avaliar um modelo específico
            results = evaluate_final_model_agnews(
                model_name=args.model,
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
            )

            logger.info(f"\n✓ Phase 2 completed for {args.model}")
            logger.info(f"  Test accuracy: {results['metrics']['test_accuracy']:.4f}")
            logger.info(f"  Test F1 (macro): {results['metrics']['test_f1_macro']:.4f}")

    # Finalização
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nTo view results in MLflow UI:")
    logger.info("  mlflow ui")
    logger.info("  Open: http://localhost:5000")
    logger.info("\nExperiments:")
    logger.info("  - hyperparameter-tuning-agnews (Phase 1)")
    logger.info("  - final-evaluation-agnews (Phase 2)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nError: {e}", exc_info=True)
        sys.exit(1)
