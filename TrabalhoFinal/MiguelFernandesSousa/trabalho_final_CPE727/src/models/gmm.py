"""
Gaussian Mixture Model (GMM) para Classificação

Implementação de classificador generativo usando modelos de mistura de Gaussianas.

Modelo:
    P(x|y=k) = Σⱼ πₖⱼ N(x|μₖⱼ, Σₖⱼ)

onde:
    - πₖⱼ: peso da componente j na classe k
    - N(x|μₖⱼ, Σₖⱼ): distribuição Gaussiana multivariada
    - Cada classe k tem seu próprio GMM com múltiplas componentes

Treinamento:
    - Algoritmo EM (Expectation-Maximization)
    - E-step: Calcula responsabilidades P(componente|x)
    - M-step: Atualiza parâmetros μ, Σ, π

Vantagens sobre Naive Bayes:
    - Captura distribuições multimodais (múltiplas "modas" por classe)
    - Matriz de covariância completa (não assume independência)
    - Mais flexível para modelar dados complexos

Classificação:
    P(y=k|x) ∝ P(y=k) * P(x|y=k)
    ŷ = argmax_k P(y=k|x)
"""
from typing import Literal, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin


class GMMClassifier(BaseEstimator, ClassifierMixin):
    """
    Classificador baseado em Gaussian Mixture Models

    Treina um GMM separado para cada classe, permitindo modelar
    distribuições multimodais e correlações entre features.

    Parâmetros:
        n_components: Número de componentes Gaussianas por classe
        covariance_type: Tipo de matriz de covariância
        max_iter: Número máximo de iterações do EM
        random_state: Seed para reprodutibilidade
        verbose: Nível de verbosidade
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        max_iter: int = 100,
        n_init: int = 10,
        random_state: int = 42,
        verbose: int = 0,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
    ):
        """
        Inicializa GMM Classifier

        Args:
            n_components: Number of Gaussian components per class (default: 2)
            covariance_type: Type of covariance matrix (default: 'full')
                - 'full': cada componente tem sua própria matriz geral
                - 'tied': todas as componentes compartilham a mesma matriz
                - 'diag': matriz diagonal (independência entre features)
                - 'spherical': matriz diagonal com variância única
            max_iter: Maximum EM iterations (default: 100)
            n_init: Number of initializations (default: 10)
            random_state: Random seed (default: 42)
            verbose: Verbosity level (default: 0)
            tol: Convergence tolerance (default: 1e-3)
            reg_covar: Regularization for covariance matrix (default: 1e-6)
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.tol = tol
        self.reg_covar = reg_covar
        self.gmms = {}  # Um GMM por classe
        self.classes_ = None
        self.class_priors_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina um GMM para cada classe

        Para cada classe k:
            1. Separa dados: X_k = {x | y = k}
            2. Treina GMM usando algoritmo EM
            3. Estima P(y=k) = n_k / n_total

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            self
        """
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        n_samples = len(y)

        # Calcular priors P(y=k)
        self.class_priors_ = {}

        # Treinar um GMM para cada classe
        for class_label in self.classes_:
            # Filtrar dados da classe
            X_class = X[y == class_label]

            # Prior P(y=k)
            self.class_priors_[class_label] = len(X_class) / n_samples

            # Criar e treinar GMM
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                verbose=self.verbose,
                tol=self.tol,
                reg_covar=self.reg_covar,
            )

            gmm.fit(X_class)
            self.gmms[class_label] = gmm

            if self.verbose > 0:
                print(
                    f"Classe {class_label}: {len(X_class)} amostras, "
                    f"converged={gmm.converged_}, n_iter={gmm.n_iter_}"
                )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes usando regra de Bayes

        Para cada amostra x:
            P(y=k|x) ∝ P(y=k) * P(x|y=k)
            onde P(x|y=k) é calculado pelo GMM da classe k

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predições (n_samples,)
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula probabilidades P(y=k|x) para cada classe

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilidades (n_samples, n_classes)
        """
        log_proba = self.predict_log_proba(X)
        # Converter log-prob para prob usando log-sum-exp trick
        proba = np.exp(log_proba - np.max(log_proba, axis=1, keepdims=True))
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula log-probabilidades log P(y=k|x)

        log P(y=k|x) = log P(y=k) + log P(x|y=k) + const

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Log-probabilidades (n_samples, n_classes)
        """
        if not self.gmms:
            raise ValueError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, len(self.classes_)))

        for idx, class_label in enumerate(self.classes_):
            gmm = self.gmms[class_label]
            prior = self.class_priors_[class_label]

            # log P(y=k|x) ∝ log P(y=k) + log P(x|y=k)
            log_likelihood = gmm.score_samples(X)  # log P(x|y=k)
            log_proba[:, idx] = np.log(prior) + log_likelihood

        return log_proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula acurácia do modelo

        Args:
            X: Features (n_samples, n_features)
            y: True labels (n_samples,)

        Returns:
            Acurácia (0 a 1)
        """
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep: bool = True) -> dict:
        """
        Retorna parâmetros do modelo

        Args:
            deep: Se True, retorna parâmetros de sub-objetos

        Returns:
            Dicionário com parâmetros
        """
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
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
        Retorna informações sobre os GMMs treinados

        Returns:
            Dicionário com informações dos modelos
        """
        if not self.gmms:
            return {"status": "not_fitted"}

        info = {
            "n_classes": len(self.classes_),
            "n_features": self.n_features_in_,
            "n_components_per_class": self.n_components,
            "covariance_type": self.covariance_type,
            "class_priors": {int(k): float(v) for k, v in self.class_priors_.items()},
            "convergence": {},
        }

        # Informações de convergência por classe
        for class_label, gmm in self.gmms.items():
            info["convergence"][int(class_label)] = {
                "converged": bool(gmm.converged_),
                "n_iter": int(gmm.n_iter_),
            }

        return info


if __name__ == "__main__":
    # Teste do modelo
    print("Testando Gaussian Mixture Model Classifier...\n")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Criar dataset de teste com distribuições multimodais
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,  # Múltiplos clusters por classe
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Treinar modelo
    print("Treinando GMM com 2 componentes por classe...")
    model = GMMClassifier(
        n_components=2, covariance_type="full", max_iter=100, verbose=1
    )
    model.fit(X_train, y_train)

    # Informações do modelo
    info = model.get_model_info()
    print(f"\nInformações do modelo:")
    print(f"  Classes: {info['n_classes']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Componentes por classe: {info['n_components_per_class']}")
    print(f"  Tipo de covariância: {info['covariance_type']}")
    print(f"\nPriors das classes:")
    for class_label, prior in info["class_priors"].items():
        conv_info = info["convergence"][class_label]
        print(
            f"  Classe {class_label}: prior={prior:.4f}, "
            f"converged={conv_info['converged']}, iters={conv_info['n_iter']}"
        )

    # Avaliar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✓ Acurácia: {acc:.4f}")

    # Testar probabilidades
    probs = model.predict_proba(X_test[:5])
    print(f"\n✓ Probabilidades (primeiras 5 amostras):")
    print(probs)
    print(f"  Soma das probabilidades: {probs.sum(axis=1)}")  # Deve ser ~1.0

    print("\n✓ GMM Classifier implementado com sucesso!")
