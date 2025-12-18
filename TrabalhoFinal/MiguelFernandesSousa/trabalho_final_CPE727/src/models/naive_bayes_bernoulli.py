"""
Classificador Naive Bayes Bernoulli

Implementação de um classificador generativo baseado no Teorema de Bayes
com suposição de features binárias ou multinomiais binarizadas.

Modelo:
    P(y|x) ∝ P(y) * P(x|y)
    P(x|y) = ∏ᵢ P(xᵢ|y)  (independência condicional)

Cada P(xᵢ|y) é modelado como uma distribuição de Bernoulli:
    P(xᵢ=1|y) = θᵢʸ
    P(xᵢ=0|y) = 1 - θᵢʸ

Adequado para features binárias ou contagens binarizadas (eg. presença/ausência de palavras).
"""
from typing import Optional

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayesBernoulli(BaseEstimator, ClassifierMixin):
    """
    Wrapper para Bernoulli Naive Bayes do scikit-learn

    Mantém compatibilidade com a interface do projeto enquanto usa
    implementação otimizada do sklearn.

    Adequado para features binárias ou binarizadas. Especialmente útil
    para classificação de texto com bag-of-words binarizado.

    Parâmetros:
        alpha: Smoothing parameter (Laplace/Lidstone smoothing)
        binarize: Threshold para binarização de features (None = não binariza)
        fit_prior: Se deve aprender probabilidades a priori das classes
        class_prior: Probabilidades a priori das classes (None = uniforme)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None,
    ):
        """
        Inicializa o classificador Naive Bayes Bernoulli

        Args:
            alpha: Smoothing parameter (default: 1.0 = Laplace smoothing)
            binarize: Threshold for binarization (default: 0.0)
                     None = no binarization
            fit_prior: Whether to learn class priors (default: True)
            class_prior: Prior probabilities (None for uniform)
        """
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o modelo Bernoulli Naive Bayes

        Para cada classe c e feature i:
            1. Calcula P(y=c) = prior
            2. Calcula θᵢᶜ = P(xᵢ=1|y=c) com Laplace smoothing:
               θᵢᶜ = (Nᵢᶜ + α) / (Nᶜ + 2α)
               onde Nᵢᶜ = número de amostras com xᵢ=1 e y=c
                    Nᶜ = número de amostras com y=c

        Args:
            X: Features (n_samples, n_features)
               Pode conter valores contínuos se binarize != None
            y: Labels (n_samples,)

        Returns:
            self
        """
        self.model = BernoulliNB(
            alpha=self.alpha,
            binarize=self.binarize,
            fit_prior=self.fit_prior,
            class_prior=self.class_prior,
        )
        self.model.fit(X, y)

        # Armazenar informações úteis
        self.classes_ = self.model.classes_
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para novos dados

        Para cada amostra x (binarizada se binarize != None):
            ŷ = argmax_c [log P(y=c) + Σᵢ (xᵢ log θᵢᶜ + (1-xᵢ) log(1-θᵢᶜ))]

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
            "alpha": self.alpha,
            "binarize": self.binarize,
            "fit_prior": self.fit_prior,
            "class_prior": self.class_prior,
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
    print("Testando Naive Bayes Bernoulli...\n")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Criar dataset de teste com features binárias
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=4,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )

    # Binarizar features (simular bag-of-words)
    X = (X > 0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Treinar modelo
    print("Treinando modelo...")
    model = NaiveBayesBernoulli(alpha=1.0, binarize=None)
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

    print("\n✓ Naive Bayes Bernoulli implementado com sucesso!")
