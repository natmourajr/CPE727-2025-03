"""
Regressão Logística One-vs-Rest (OvR)

Implementação de classificador discriminativo multiclasse usando estratégia OvR.

Estratégia:
    - Treina K classificadores binários independentes (um para cada classe)
    - Cada classificador k aprende: "classe k vs todas as outras"
    - Predição: argmax_k P(y=k|x) entre todos os K classificadores

Modelo para cada classificador binário:
    P(y=k|x) = σ(wₖᵀx + bₖ)
    onde σ(z) = 1/(1 + exp(-z)) é a função sigmoid

Vantagens:
    - Simples de implementar e paralelizar
    - Cada classificador é independente

Desvantagens:
    - K modelos separados (mais parâmetros que Softmax)
    - Não aprende relações entre classes
    - Probabilidades não calibradas naturalmente
"""
from typing import Optional, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionOvR(BaseEstimator, ClassifierMixin):
    """
    Regressão Logística One-vs-Rest

    Treina K classificadores binários, um para cada classe.
    Cada classificador distingue uma classe de todas as outras.

    Parâmetros:
        C: Inverso da força de regularização (maior = menos regularização)
        solver: Algoritmo de otimização
        max_iter: Número máximo de iterações
        random_state: Seed para reprodutibilidade
        verbose: Nível de verbosidade
    """

    def __init__(
        self,
        C: float = 1.0,
        solver: Literal["lbfgs", "liblinear", "newton-cg", "sag", "saga"] = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        verbose: int = 0,
        tol: float = 1e-4,
    ):
        """
        Inicializa Regressão Logística OvR

        Args:
            C: Inverse of regularization strength (default: 1.0)
            solver: Optimization algorithm (default: 'lbfgs')
            max_iter: Maximum iterations (default: 1000)
            random_state: Random seed (default: 42)
            verbose: Verbosity level (default: 0)
            tol: Tolerance for stopping criteria (default: 1e-4)
        """
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.tol = tol
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina K classificadores binários One-vs-Rest

        Para cada classe k:
            1. Cria labels binários: y_binary = (y == k)
            2. Treina classificador: min_wₖ [ -Σᵢ log P(y_binary|xᵢ) + (1/2C)||wₖ||² ]

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            self
        """
        # multi_class='ovr' força estratégia One-vs-Rest
        # penalty='l2' adiciona regularização Ridge
        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class="ovr",  # One-vs-Rest strategy
            penalty="l2",  # Regularização L2
            random_state=self.random_state,
            verbose=self.verbose,
            tol=self.tol,
            n_jobs=-1,  # Paraleliza treinamento dos K modelos
        )

        self.model.fit(X, y)

        # Armazenar informações
        self.classes_ = self.model.classes_
        self.n_features_in_ = X.shape[1]
        self.coef_ = self.model.coef_  # Shape: (n_classes, n_features)
        self.intercept_ = self.model.intercept_  # Shape: (n_classes,)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes usando argmax dos scores OvR

        Para cada amostra x:
            1. Calcula score de cada classificador k: sₖ(x) = wₖᵀx + bₖ
            2. Predição: ŷ = argmax_k sₖ(x)

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predições (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula probabilidades usando sigmoid de cada classificador OvR

        NOTA: Probabilidades não são perfeitamente calibradas em OvR
              (soma pode não ser exatamente 1.0, mas sklearn normaliza)

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilidades (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula log-probabilidades

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Log-probabilidades (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_log_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula acurácia do modelo

        Args:
            X: Features (n_samples, n_features)
            y: True labels (n_samples,)

        Returns:
            Acurácia (0 a 1)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")

        return self.model.score(X, y)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula scores de decisão de cada classificador OvR

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Decision scores (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before decision function")

        return self.model.decision_function(X)

    def get_params(self, deep: bool = True) -> dict:
        """
        Retorna parâmetros do modelo

        Args:
            deep: Se True, retorna parâmetros de sub-objetos

        Returns:
            Dicionário com parâmetros
        """
        return {
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "tol": self.tol,
        }

    def set_params(self, **params):
        """
        Define parâmetros do modelo

        Args:
            **params: Parâmetros a serem definidos

        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo treinado

        Returns:
            Dicionário com informações do modelo
        """
        if self.model is None:
            return {"status": "not_fitted"}

        # Em OvR, temos K modelos binários
        n_binary_models = len(self.classes_)
        params_per_model = self.n_features_in_ + 1  # pesos + bias

        return {
            "n_classes": len(self.classes_),
            "n_features": self.n_features_in_,
            "n_binary_classifiers": n_binary_models,
            "n_parameters_total": self.coef_.size + self.intercept_.size,
            "n_parameters_per_classifier": params_per_model,
            "n_iterations": self.model.n_iter_.tolist()
            if hasattr(self.model, "n_iter_")
            else None,
            "classes": self.classes_.tolist(),
        }


if __name__ == "__main__":
    # Teste do modelo
    print("Testando Regressão Logística One-vs-Rest (OvR)...\n")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Criar dataset de teste
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=10,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Treinar modelo
    print("Treinando modelo OvR...")
    model = LogisticRegressionOvR(C=1.0, max_iter=1000, verbose=0)
    model.fit(X_train, y_train)

    # Informações do modelo
    info = model.get_model_info()
    print(f"\nInformações do modelo:")
    print(f"  Classes: {info['n_classes']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Classificadores binários: {info['n_binary_classifiers']}")
    print(f"  Parâmetros totais: {info['n_parameters_total']}")
    print(f"  Parâmetros por classificador: {info['n_parameters_per_classifier']}")
    if info["n_iterations"]:
        print(f"  Iterações por classificador: {info['n_iterations']}")

    # Avaliar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✓ Acurácia: {acc:.4f}")

    # Testar probabilidades
    probs = model.predict_proba(X_test[:5])
    print(f"\n✓ Probabilidades (primeiras 5 amostras):")
    print(probs)
    print(f"  Soma das probabilidades: {probs.sum(axis=1)}")  # Deve ser ~1.0

    # Comparar decision function
    scores = model.decision_function(X_test[:3])
    print(f"\n✓ Decision scores (primeiras 3 amostras):")
    print(scores)

    print("\n✓ Logistic Regression OvR implementado com sucesso!")
