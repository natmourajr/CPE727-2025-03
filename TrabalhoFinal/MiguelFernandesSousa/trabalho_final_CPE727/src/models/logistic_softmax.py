"""
Regressão Logística Multinomial (Softmax)

Implementação de classificador discriminativo multiclasse usando função softmax.

Modelo:
    P(y=k|x) = exp(wₖᵀx + bₖ) / Σⱼ exp(wⱼᵀx + bⱼ)

onde wₖ são os pesos para a classe k.

Função de perda:
    L = -Σᵢ Σₖ yᵢₖ log(P(y=k|xᵢ))  (Cross-Entropy)

Otimização:
    - Gradiente descendente ou métodos quasi-Newton (L-BFGS, Newton-CG)
    - Regularização L2 (Ridge): ||w||² para evitar overfitting
"""
from typing import Optional, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionSoftmax(BaseEstimator, ClassifierMixin):
    """
    Regressão Logística Multinomial com Softmax

    Aprende todas as K classes simultaneamente através da função softmax.
    Mais eficiente que OvR pois treina um único modelo multiclasse.

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
        solver: Literal["lbfgs", "newton-cg", "sag", "saga"] = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        verbose: int = 0,
        tol: float = 1e-4,
    ):
        """
        Inicializa Regressão Logística Softmax

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
        Treina o modelo de Regressão Logística Multinomial

        Otimiza:
            min_w [ -Σᵢ Σₖ yᵢₖ log(softmax(Wx)ᵢₖ) + (1/2C)||W||² ]

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            self
        """
        # multi_class='multinomial' força uso de softmax nativo
        # penalty='l2' adiciona regularização Ridge
        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class="multinomial",  # Softmax nativo
            penalty="l2",  # Regularização L2
            random_state=self.random_state,
            verbose=self.verbose,
            tol=self.tol,
            n_jobs=-1,  # Usar todos os cores
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
        Prediz classes usando argmax das probabilidades softmax

        ŷ = argmax_k P(y=k|x)

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
        Calcula probabilidades via softmax

        P(y=k|x) = exp(wₖᵀx + bₖ) / Σⱼ exp(wⱼᵀx + bⱼ)

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
        Calcula scores de decisão (logits antes do softmax)

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

        return {
            "n_classes": len(self.classes_),
            "n_features": self.n_features_in_,
            "n_parameters": self.coef_.size + self.intercept_.size,
            "n_iterations": self.model.n_iter_[0] if hasattr(self.model, "n_iter_") else None,
            "classes": self.classes_.tolist(),
        }


if __name__ == "__main__":
    # Teste do modelo
    print("Testando Regressão Logística Softmax (Multinomial)...\n")

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
    print("Treinando modelo...")
    model = LogisticRegressionSoftmax(C=1.0, max_iter=1000, verbose=1)
    model.fit(X_train, y_train)

    # Informações do modelo
    info = model.get_model_info()
    print(f"\nInformações do modelo:")
    print(f"  Classes: {info['n_classes']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Parâmetros totais: {info['n_parameters']}")
    print(f"  Iterações: {info['n_iterations']}")

    # Avaliar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✓ Acurácia: {acc:.4f}")

    # Testar probabilidades
    probs = model.predict_proba(X_test[:5])
    print(f"\n✓ Probabilidades (primeiras 5 amostras):")
    print(probs)
    print(f"  Soma das probabilidades: {probs.sum(axis=1)}")  # Deve ser ~1.0

    print("\n✓ Logistic Regression Softmax implementado com sucesso!")
