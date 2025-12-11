"""
Classificador Naive Bayes Multinomial

Implementação de um classificador generativo baseado no Teorema de Bayes
com suposição de features seguindo distribuição multinomial.

Modelo:
    P(y|x) ∝ P(y) * P(x|y)
    P(x|y) = ∏ᵢ P(xᵢ|y)  (independência condicional)

Cada P(xᵢ|y) é modelado como uma distribuição multinomial:
    P(xᵢ|y) = θᵢʸ^xᵢ

Adequado para features de contagem (eg. frequência de palavras, TF-IDF).
IMPORTANTE: Requer features não-negativas (≥ 0).
"""
from typing import Optional

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, ClassifierMixin


class NaiveBayesMultinomial(BaseEstimator, ClassifierMixin):
    """
    Wrapper para Multinomial Naive Bayes do scikit-learn

    Mantém compatibilidade com a interface do projeto enquanto usa
    implementação otimizada do sklearn.

    Adequado para features de contagem ou frequências. Especialmente útil
    para classificação de texto com TF-IDF ou bag-of-words.

    IMPORTANTE: Requer que todas as features sejam não-negativas.
    Para Fashion MNIST, normalização deve ser [0, 1] em vez de [-1, 1].

    Parâmetros:
        alpha: Smoothing parameter (Laplace/Lidstone smoothing)
        fit_prior: Se deve aprender probabilidades a priori das classes
        class_prior: Probabilidades a priori das classes (None = uniforme)
        force_alpha: Se True, força alpha > 0 (recomendado)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None,
        force_alpha: bool = True,
    ):
        """
        Inicializa o classificador Naive Bayes Multinomial

        Args:
            alpha: Smoothing parameter (default: 1.0 = Laplace smoothing)
                  Valores comuns: 0.1, 0.5, 1.0
            fit_prior: Whether to learn class priors (default: True)
            class_prior: Prior probabilities (None for uniform)
            force_alpha: Force alpha > 0 for numerical stability (default: True)
        """
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.force_alpha = force_alpha
        self.model = None

        # Validação
        if self.force_alpha and self.alpha <= 0:
            raise ValueError(
                f"alpha must be > 0 for MultinomialNB, got {self.alpha}. "
                "Set force_alpha=False to disable this check."
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o modelo Multinomial Naive Bayes

        Para cada classe c e feature i:
            1. Calcula P(y=c) = prior
            2. Calcula θᵢᶜ = P(xᵢ|y=c) com Laplace smoothing:
               θᵢᶜ = (Nᵢᶜ + α) / (Nᶜ + nα)
               onde Nᵢᶜ = soma de xᵢ para amostras com y=c
                    Nᶜ = soma de todas as features para amostras com y=c
                    n = número de features

        Args:
            X: Features (n_samples, n_features)
               IMPORTANTE: Deve conter apenas valores não-negativos (≥ 0)
            y: Labels (n_samples,)

        Returns:
            self

        Raises:
            ValueError: Se X contém valores negativos
        """
        # Verificar valores não-negativos
        if np.any(X < 0):
            raise ValueError(
                "MultinomialNB requires non-negative features. "
                "For Fashion MNIST, use normalization_range='0_1' instead of '-1_1'."
            )

        self.model = MultinomialNB(
            alpha=self.alpha, fit_prior=self.fit_prior, class_prior=self.class_prior
        )
        self.model.fit(X, y)

        # Armazenar informações úteis
        self.classes_ = self.model.classes_
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para novos dados

        Para cada amostra x:
            ŷ = argmax_c [log P(y=c) + Σᵢ xᵢ log θᵢᶜ]

        Args:
            X: Features (n_samples, n_features)
               IMPORTANTE: Deve conter apenas valores não-negativos (≥ 0)

        Returns:
            Predições (n_samples,)

        Raises:
            ValueError: Se X contém valores negativos
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")

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

        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")

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

        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")

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

        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features")

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
            "fit_prior": self.fit_prior,
            "class_prior": self.class_prior,
            "force_alpha": self.force_alpha,
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
    print("Testando Naive Bayes Multinomial...\n")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler

    # Criar dataset de teste
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=4,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )

    # Garantir features não-negativas (simular TF-IDF)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Treinar modelo
    print("Treinando modelo...")
    model = NaiveBayesMultinomial(alpha=1.0)
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

    # Testar validação de valores negativos
    print("\n✓ Testando validação de valores negativos...")
    try:
        X_negative = np.array([[-1, 0, 1]])
        model.predict(X_negative)
        print("✗ ERRO: Deveria ter lançado ValueError")
    except ValueError as e:
        print(f"✓ Validação funcionando: {e}")

    print("\n✓ Naive Bayes Multinomial implementado com sucesso!")
