"""
PCA Transformer para redução de dimensionalidade

Implementa transformação PCA (Principal Component Analysis) para reduzir
a dimensionalidade dos dados enquanto preserva a maior parte da variância.

PCA projeta os dados em um subespaço de menor dimensão encontrando as
direções (componentes principais) de máxima variância.

Características:
    - Determinístico (mesmos dados → mesma transformação)
    - Permite extensão out-of-sample (transformar novos dados)
    - Preserva informação de variância explicada
    - Adequado para training de modelos (diferente de t-SNE)
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.decomposition import PCA
import pickle


class PCATransformer:
    """
    Wrapper para PCA do scikit-learn com rastreamento de variância

    Transforma dados de alta dimensão (eg. 784 features de Fashion MNIST)
    para dimensão reduzida (eg. 2, 3, 10, 50, 100 componentes) enquanto
    preserva informação máxima de variância.

    Parâmetros:
        n_components: Número de componentes principais (2, 3, 10, 50, ou 100)
        random_state: Seed para reprodutibilidade
        whiten: Se True, normaliza componentes para variância unitária
    """

    def __init__(
        self, n_components: int, random_state: int = 42, whiten: bool = False
    ):
        """
        Inicializa o PCA transformer

        Args:
            n_components: Number of principal components to keep
                         Recommended values based on EDA:
                         - 2: ~46.8% variance (visualização)
                         - 3: ~52-55% variance (visualização 3D)
                         - 10: ~72.0% variance (baseline reduzido)
                         - 50: ~86.3% variance (bom tradeoff)
                         - 100: ~91.2% variance (alta fidelidade)
            random_state: Random seed for reproducibility (default: 42)
            whiten: Whether to whiten components (default: False)
                   Whitening escala componentes para variância unitária
        """
        self.n_components = n_components
        self.random_state = random_state
        self.whiten = whiten

        self.pca = None
        self.variance_explained = None
        self.cumulative_variance = None
        self.mean_ = None
        self.components_ = None

    def fit(self, X_train: np.ndarray) -> "PCATransformer":
        """
        Fit PCA apenas nos dados de treino

        Calcula:
            1. Média μ dos dados de treino
            2. Matriz de covariância Σ = (1/n) X^T X
            3. Autovalores λ e autovetores v de Σ
            4. Seleciona top-k autovetores como componentes principais

        Args:
            X_train: Training features (n_samples, n_features)

        Returns:
            self
        """
        print(
            f"\nFitting PCA with {self.n_components} components on {X_train.shape[0]} samples..."
        )

        self.pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            whiten=self.whiten,
        )
        self.pca.fit(X_train)

        # Armazenar informações de variância
        self.variance_explained = self.pca.explained_variance_ratio_ * 100
        self.cumulative_variance = np.cumsum(self.variance_explained)

        # Armazenar componentes e média
        self.mean_ = self.pca.mean_
        self.components_ = self.pca.components_

        print(f"  ✓ PCA fitted")
        print(
            f"  ✓ Variance explained by {self.n_components} components: {self.cumulative_variance[-1]:.2f}%"
        )
        print(f"  ✓ Top 5 components explain:")
        for i in range(min(5, self.n_components)):
            print(
                f"      PC{i+1}: {self.variance_explained[i]:.2f}% (cumulative: {self.cumulative_variance[i]:.2f}%)"
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforma dados para o espaço PCA

        Projeção: X_pca = (X - μ) @ V^T
        onde V são os componentes principais (autovetores)

        Args:
            X: Features (n_samples, n_features)

        Returns:
            X_pca: Transformed features (n_samples, n_components)
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before transform")

        return self.pca.transform(X)

    def fit_transform(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Fit PCA no treino e transforma todos os conjuntos

        IMPORTANTE: Fit apenas no treino para evitar data leakage

        Args:
            X_train: Training features (n_samples, n_features)
            X_val: Validation features (optional)
            X_test: Test features (optional)

        Returns:
            X_train_pca, X_val_pca, X_test_pca
        """
        # Fit apenas no treino
        self.fit(X_train)

        # Transform todos os conjuntos
        X_train_pca = self.transform(X_train)
        X_val_pca = self.transform(X_val) if X_val is not None else None
        X_test_pca = self.transform(X_test) if X_test is not None else None

        print(f"\n  Transformed shapes:")
        print(f"    X_train: {X_train.shape} → {X_train_pca.shape}")
        if X_val_pca is not None:
            print(f"    X_val: {X_val.shape} → {X_val_pca.shape}")
        if X_test_pca is not None:
            print(f"    X_test: {X_test.shape} → {X_test_pca.shape}")

        return X_train_pca, X_val_pca, X_test_pca

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Reconstrui dados originais a partir do espaço PCA

        Reconstrução: X_reconst = X_pca @ V + μ

        Útil para visualizar o que foi perdido na redução de dimensionalidade.

        Args:
            X_pca: Transformed features (n_samples, n_components)

        Returns:
            X_reconst: Reconstructed features (n_samples, n_features)
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before inverse_transform")

        return self.pca.inverse_transform(X_pca)

    def get_variance_info(self) -> dict:
        """
        Retorna informações detalhadas sobre variância explicada

        Returns:
            dict com variance_per_component, cumulative_variance, total_variance
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted first")

        return {
            "variance_per_component": self.variance_explained,
            "cumulative_variance": self.cumulative_variance,
            "total_variance": self.cumulative_variance[-1],
            "n_components": self.n_components,
        }

    def save(self, filepath: Path):
        """
        Salva PCA transformer para disco

        Args:
            filepath: Path to save file (.pkl)
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        print(f"  ✓ PCA transformer saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> "PCATransformer":
        """
        Carrega PCA transformer do disco

        Args:
            filepath: Path to saved file (.pkl)

        Returns:
            PCATransformer instance
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            transformer = pickle.load(f)

        print(
            f"  ✓ PCA transformer loaded from {filepath} ({transformer.n_components} components)"
        )
        return transformer


if __name__ == "__main__":
    # Teste do PCA transformer
    print("Testando PCA Transformer...\n")

    from sklearn.datasets import make_classification

    # Criar dataset de teste (simular Fashion MNIST)
    X, y = make_classification(
        n_samples=1000,
        n_features=784,  # Simular Fashion MNIST
        n_informative=100,
        n_redundant=50,
        n_classes=10,
        random_state=42,
    )

    # Split train/val/test
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Dataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")

    # Testar diferentes números de componentes
    for n_components in [2, 3, 10, 50, 100]:
        print(f"\n{'='*60}")
        print(f"Testing PCA with {n_components} components")
        print(f"{'='*60}")

        # Criar e treinar PCA
        pca_transformer = PCATransformer(n_components=n_components, random_state=42)

        # Fit e transform
        X_train_pca, X_val_pca, X_test_pca = pca_transformer.fit_transform(
            X_train, X_val, X_test
        )

        # Variância
        variance_info = pca_transformer.get_variance_info()
        print(f"\n  Variance info:")
        print(f"    Total variance: {variance_info['total_variance']:.2f}%")

        # Testar reconstrução
        X_train_reconst = pca_transformer.inverse_transform(X_train_pca)
        reconstruction_error = np.mean((X_train - X_train_reconst) ** 2)
        print(
            f"    Reconstruction MSE: {reconstruction_error:.6f} (lower = better fidelity)"
        )

    # Testar salvar/carregar
    print(f"\n{'='*60}")
    print("Testing save/load")
    print(f"{'='*60}")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    pca_transformer.save(tmp_path)
    loaded_transformer = PCATransformer.load(tmp_path)

    # Verificar se é o mesmo
    X_test_pca_original = pca_transformer.transform(X_test)
    X_test_pca_loaded = loaded_transformer.transform(X_test)
    assert np.allclose(X_test_pca_original, X_test_pca_loaded)
    print("  ✓ Save/load working correctly")

    # Limpar
    tmp_path.unlink()

    print("\n✓ PCA Transformer implementado com sucesso!")
