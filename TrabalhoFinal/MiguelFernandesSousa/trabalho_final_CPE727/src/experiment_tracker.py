"""
Experiment Tracker usando MLflow

Este módulo fornece uma interface simplificada para rastrear experimentos
de machine learning usando MLflow.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CLASS_NAMES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PLOTS_DIR,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


class ExperimentTracker:
    """
    Wrapper para MLflow que facilita o rastreamento de experimentos
    """

    def __init__(self, experiment_name: str = MLFLOW_EXPERIMENT_NAME):
        """
        Inicializa o tracker

        Args:
            experiment_name: Nome do experimento MLflow
        """
        self.experiment_name = experiment_name

        # Configurar MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        logger.info(f"Experiment: {experiment_name}")

    def start_run(self, run_name: str, tags: Optional[Dict[str, Any]] = None):
        """
        Inicia um novo run do MLflow

        Args:
            run_name: Nome do run
            tags: Tags adicionais (opcional)

        Returns:
            Context manager do MLflow run
        """
        tags = tags or {}
        tags["timestamp"] = datetime.now().isoformat()

        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]):
        """
        Loga parâmetros do modelo

        Args:
            params: Dicionário com parâmetros
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Loga métricas

        Args:
            metrics: Dicionário com métricas
            step: Step/época (opcional)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, artifact_path: str = "model"):
        """
        Salva modelo no MLflow

        Args:
            model: Modelo sklearn/torch
            artifact_path: Caminho do artefato
        """
        try:
            mlflow.sklearn.log_model(model, artifact_path)
            logger.info(f"Modelo salvo em: {artifact_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Loga artefato (arquivo)

        Args:
            local_path: Caminho local do arquivo
            artifact_path: Caminho no MLflow (opcional)
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_figure(self, fig, filename: str):
        """
        Salva e loga uma figura matplotlib

        Args:
            fig: Figura matplotlib
            filename: Nome do arquivo
        """
        filepath = PLOTS_DIR / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(str(filepath))
        logger.info(f"Figura salva: {filename}")

    def track_training(
        self,
        model_name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
        val_split: Optional[tuple] = None,
    ) -> str:
        """
        Rastreia um experimento completo de treinamento

        Args:
            model_name: Nome do modelo
            model: Instância do modelo
            X_train: Features de treino
            y_train: Labels de treino
            X_test: Features de teste
            y_test: Labels de teste
            model_params: Parâmetros do modelo (opcional)
            val_split: Tupla (X_val, y_val) para validação (opcional)

        Returns:
            run_id: ID do run no MLflow
        """
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self.start_run(run_name=run_name, tags={"model_type": model_name}):
            # Logar parâmetros do dataset
            self.log_params(
                {
                    "model_name": model_name,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "n_features": X_train.shape[1],
                    "n_classes": len(np.unique(y_train)),
                }
            )

            # Logar parâmetros do modelo
            if model_params:
                self.log_params(model_params)

            # Treinar e medir tempo
            logger.info(f"Treinando {model_name}...")
            start_time = time.time()

            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                # Predições
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Métricas de treino
                train_metrics = self._compute_metrics(y_train, y_train_pred, "train")
                train_metrics["train_time"] = training_time

                # Métricas de teste
                test_metrics = self._compute_metrics(y_test, y_test_pred, "test")

                # Tempo de inferência
                start_time = time.time()
                _ = model.predict(X_test[:100])  # Batch pequeno para medir
                inference_time = (time.time() - start_time) / 100  # Por amostra
                test_metrics["inference_time_per_sample"] = inference_time

                # Métricas de validação (se fornecido)
                if val_split is not None:
                    X_val, y_val = val_split
                    y_val_pred = model.predict(X_val)
                    val_metrics = self._compute_metrics(y_val, y_val_pred, "val")
                    self.log_metrics(val_metrics)

                # Logar todas as métricas
                self.log_metrics({**train_metrics, **test_metrics})

                # Salvar modelo
                self.log_model(model)

                # Logar confusion matrix
                self._log_confusion_matrix(y_test, y_test_pred, model_name)

                # Logar classification report
                self._log_classification_report(y_test, y_test_pred, model_name)

                logger.info(f"✓ {model_name} treinado com sucesso!")
                logger.info(f"  Acurácia (teste): {test_metrics['test_accuracy']:.4f}")
                logger.info(f"  Tempo de treino: {training_time:.2f}s")

            except Exception as e:
                logger.error(f"Erro ao treinar {model_name}: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                raise

            return mlflow.active_run().info.run_id

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calcula métricas de classificação

        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            prefix: Prefixo para as métricas (train/val/test)

        Returns:
            Dicionário com métricas
        """
        prefix = f"{prefix}_" if prefix else ""

        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            f"{prefix}recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            f"{prefix}precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            f"{prefix}recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            f"{prefix}f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        return metrics

    def _log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        class_names: list = None,
    ):
        """
        Cria e loga matriz de confusão

        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            model_name: Nome do modelo
            class_names: Lista de nomes das classes (default: CLASS_NAMES do Fashion MNIST)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            labels = class_names if class_names is not None else CLASS_NAMES
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.title(f"Confusion Matrix - {model_name}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            filename = f"confusion_matrix_{model_name}.png"
            self.log_figure(plt.gcf(), filename)
            plt.close()

        except Exception as e:
            logger.warning(f"Erro ao criar confusion matrix: {e}")

    def _log_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        class_names: list = None,
    ):
        """
        Cria e loga classification report

        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            model_name: Nome do modelo
            class_names: Lista de nomes das classes (default: CLASS_NAMES do Fashion MNIST)
        """
        try:
            labels = class_names if class_names is not None else CLASS_NAMES
            report = classification_report(
                y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
            )

            # Salvar como texto
            report_text = classification_report(
                y_true, y_pred, target_names=labels, zero_division=0
            )

            filename = PLOTS_DIR / f"classification_report_{model_name}.txt"
            filename.write_text(report_text)
            mlflow.log_artifact(str(filename))

            # Logar métricas por classe
            for class_name in labels:
                if class_name in report:
                    for metric in ["precision", "recall", "f1-score"]:
                        mlflow.log_metric(
                            f"class_{class_name}_{metric}", report[class_name][metric]
                        )

        except Exception as e:
            logger.warning(f"Erro ao criar classification report: {e}")

    @staticmethod
    def get_best_run(metric: str = "test_accuracy", ascending: bool = False) -> Dict[str, Any]:
        """
        Retorna o melhor run baseado em uma métrica

        Args:
            metric: Nome da métrica
            ascending: Se True, menor é melhor

        Returns:
            Dicionário com informações do run
        """
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            logger.warning(f"Experimento {MLFLOW_EXPERIMENT_NAME} não encontrado")
            return {}

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if runs.empty:
            logger.warning("Nenhum run encontrado")
            return {}

        best_run = runs.iloc[0].to_dict()
        return best_run

    @staticmethod
    def compare_runs(metric: str = "test_accuracy") -> Any:
        """
        Compara todos os runs baseado em uma métrica

        Args:
            metric: Nome da métrica para ordenar

        Returns:
            DataFrame com runs ordenados
        """
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            logger.warning(f"Experimento {MLFLOW_EXPERIMENT_NAME} não encontrado")
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
        )

        return runs


if __name__ == "__main__":
    # Teste básico do tracker
    print("Testando ExperimentTracker...\n")

    from sklearn.datasets import make_classification
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split

    # Criar dataset de teste
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=10, n_informative=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar tracker
    tracker = ExperimentTracker()

    # Treinar modelo de teste
    model = GaussianNB()
    run_id = tracker.track_training(
        model_name="test_naive_bayes",
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_params={"var_smoothing": 1e-9},
    )

    print(f"\n✓ Run ID: {run_id}")
    print("\nPara visualizar os resultados, execute:")
    print("  mlflow ui")
    print("  Acesse: http://localhost:5000")
