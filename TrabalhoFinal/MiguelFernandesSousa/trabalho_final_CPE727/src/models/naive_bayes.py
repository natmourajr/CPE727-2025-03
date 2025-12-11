"""
Classificador Naive Bayes Gaussiano

Implementação de um classificador generativo baseado no Teorema de Bayes
com suposição de independência condicional entre features.

Modelo:
    P(y|x) ∝ P(y) * P(x|y)
    P(x|y) = ∏ᵢ P(xᵢ|y)  (independência condicional)

Cada P(xᵢ|y) é modelado como uma distribuição Gaussiana N(μᵢʸ, σᵢʸ)
com matriz de covariância diagonal (variâncias independentes por feature).
"""
from typing import Optional

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayesGaussian(BaseEstimator, ClassifierMixin):
    """
    Wrapper para Gaussian Naive Bayes do scikit-learn

    Mantém compatibilidade com a interface do projeto enquanto usa
    implementação otimizada do sklearn.

    Parâmetros:
        var_smoothing: Porção da maior variância de todas as features
                       adicionada às variâncias para estabilidade (Laplace smoothing)
        priors: Probabilidades a priori das classes (None = uniforme)
    """

    def __init__(self, var_smoothing: float = 1e-9, priors: Optional[np.ndarray] = None):
        """
        Inicializa o classificador Naive Bayes Gaussiano

        Args:
            var_smoothing: Smoothing parameter (default: 1e-9)
            priors: Prior probabilities (None for uniform)
        """
        self.var_smoothing = var_smoothing
        self.priors = priors
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o modelo Naive Bayes

        Para cada classe c:
            1. Calcula P(y=c) = prior
            2. Calcula μᵢᶜ = E[xᵢ|y=c] para cada feature i
            3. Calcula σᵢᶜ = Var[xᵢ|y=c] para cada feature i

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            self
        """
        self.model = GaussianNB(var_smoothing=self.var_smoothing, priors=self.priors)
        self.model.fit(X, y)

        # Armazenar informações úteis
        self.classes_ = self.model.classes_
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para novos dados

        Para cada amostra x:
            ŷ = argmax_c [log P(y=c) + Σᵢ log P(xᵢ|y=c)]

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
        Prediz probabilidades de classe

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
        Prediz log-probabilidades de classe

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

    def get_params(self, deep: bool = True) -> dict:
        """
        Retorna parâmetros do modelo

        Args:
            deep: Se True, retorna parâmetros de sub-objetos

        Returns:
            Dicionário com parâmetros
        """
        return {
            "var_smoothing": self.var_smoothing,
            "priors": self.priors,
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


if __name__ == "__main__":
    # Teste do modelo
    print("Testando Naive Bayes Gaussiano...\n")

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
    model = NaiveBayesGaussian(var_smoothing=1e-9)
    model.fit(X_train, y_train)

    # Avaliar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✓ Acurácia: {acc:.4f}")
    print(f"✓ Classes encontradas: {model.classes_}")
    print(f"✓ Número de features: {model.n_features_in_}")

    # Testar probabilidades
    probs = model.predict_proba(X_test[:5])
    print(f"\n✓ Probabilidades (primeiras 5 amostras):")
    print(probs)

    print("\n✓ Naive Bayes Gaussiano implementado com sucesso!")
