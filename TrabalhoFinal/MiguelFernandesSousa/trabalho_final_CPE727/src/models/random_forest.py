"""
Classificador Random Forest

Implementação de um classificador discriminativo baseado em ensemble de
árvores de decisão (CART - Classification and Regression Trees).

Modelo:
    Ensemble de T árvores de decisão treinadas em subsets aleatórios dos dados
    com features aleatórias em cada split.

    Predição: Votação majoritária (classificação) ou média (regressão)
    ŷ = mode({h₁(x), h₂(x), ..., h_T(x)})

Vantagens:
    - Não requer normalização de features
    - Robusto a outliers e features irrelevantes
    - Pode capturar relações não-lineares complexas
    - Fornece importância de features
    - Reduz overfitting através de bagging e feature randomization
"""
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomForest(BaseEstimator, ClassifierMixin):
    """
    Wrapper para Random Forest Classifier do scikit-learn

    Mantém compatibilidade com a interface do projeto enquanto usa
    implementação otimizada do sklearn.

    Random Forest combina bagging (bootstrap aggregating) com feature
    randomization para criar um ensemble robusto de árvores de decisão.

    Parâmetros:
        n_estimators: Número de árvores no ensemble
        max_depth: Profundidade máxima das árvores (None = sem limite)
        min_samples_split: Mínimo de amostras para split interno
        min_samples_leaf: Mínimo de amostras em nó folha
        max_features: Número de features a considerar em cada split
        bootstrap: Se deve usar bootstrap sampling
        max_samples: Fração de amostras para cada árvore
        class_weight: Pesos das classes (None ou 'balanced')
        random_state: Seed para reprodutibilidade
        n_jobs: Número de jobs paralelos (-1 = todos os cores)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        max_samples: Optional[float] = None,
        class_weight: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        """
        Inicializa o classificador Random Forest

        Args:
            n_estimators: Number of trees (default: 100)
                         More trees → better performance but slower
            max_depth: Maximum tree depth (default: None = unlimited)
                      Lower values → less overfitting but may underfit
            min_samples_split: Minimum samples to split (default: 2)
            min_samples_leaf: Minimum samples in leaf (default: 1)
            max_features: Features to consider per split (default: 'sqrt')
                         Options: 'sqrt', 'log2', int, float
            bootstrap: Use bootstrap sampling (default: True)
            max_samples: Fraction of samples per tree (default: None = 1.0)
            class_weight: Class weights (default: None)
                         'balanced' for imbalanced datasets
            random_state: Random seed (default: 42)
            n_jobs: Parallel jobs (default: -1 = all cores)
            verbose: Verbosity level (default: 0)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Treina o Random Forest

        Para cada árvore t=1..T:
            1. Amostra bootstrap D_t de D (n amostras com reposição)
            2. Treina árvore h_t em D_t com feature randomization:
               - Em cada split, considera apenas m features aleatórias
               - Escolhe melhor split usando critério de impureza (Gini)
            3. Árvore cresce até max_depth ou min_samples_leaf

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            self
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            max_samples=self.max_samples,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.model.fit(X, y)

        # Armazenar informações úteis
        self.classes_ = self.model.classes_
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = self.model.feature_importances_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para novos dados

        Para cada amostra x:
            1. Coleta predições de todas as T árvores: {h₁(x), ..., h_T(x)}
            2. Retorna classe mais votada (mode):
               ŷ = argmax_c Σₜ 𝟙[h_t(x) = c]

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

        Para cada amostra x e classe c:
            P(y=c|x) = (1/T) Σₜ 𝟙[h_t(x) = c]
            Proporção de árvores que votaram na classe c

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilidades (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)

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

    def get_feature_importances(self) -> np.ndarray:
        """
        Retorna importâncias das features

        Importância calculada como redução média de impureza (Gini)
        normalizada através de todas as árvores.

        Returns:
            Feature importances (n_features,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        return self.feature_importances_

    def get_params(self, deep: bool = True) -> dict:
        """
        Retorna parâmetros do modelo

        Args:
            deep: Se True, retorna parâmetros de sub-objetos

        Returns:
            Dicionário com parâmetros
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "max_samples": self.max_samples,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
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
    print("Testando Random Forest...\n")

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
    print("Treinando modelo com 100 árvores...")
    model = RandomForest(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Avaliar
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✓ Acurácia: {acc:.4f}")
    print(f"✓ Classes encontradas: {model.classes_}")
    print(f"✓ Número de features: {model.n_features_in_}")
    print(f"✓ Número de árvores: {model.n_estimators}")

    # Testar probabilidades
    probs = model.predict_proba(X_test[:5])
    print(f"\n✓ Probabilidades (primeiras 5 amostras):")
    print(probs)

    # Testar importância de features
    importances = model.get_feature_importances()
    print(f"\n✓ Top 5 features mais importantes:")
    top_indices = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. Feature {idx}: {importances[idx]:.4f}")

    print("\n✓ Random Forest implementado com sucesso!")
