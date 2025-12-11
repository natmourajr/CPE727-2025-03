"""
Configuração de hiperparâmetros para Fashion MNIST

Define os grids de hiperparâmetros específicos para Fashion MNIST.

IMPORTANTE: GMM pode usar tanto 'full' quanto 'diag' covariance para Fashion MNIST
(784 features é computacionalmente viável para full covariance).
"""

# Grid de hiperparâmetros para cada modelo no Fashion MNIST
HYPERPARAMETER_GRIDS = {
    # Naive Bayes Gaussiano
    "naive_bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],  # Laplace smoothing
    },
    # Naive Bayes Bernoulli
    "naive_bayes_bernoulli": {
        "alpha": [0.1, 0.5, 1.0, 2.0],  # Laplace smoothing
        "binarize": [0.0, 0.3, 0.5],  # Threshold para binarização
    },
    # Naive Bayes Multinomial
    # IMPORTANTE: Requer normalization_range='0_1' no data loader
    "naive_bayes_multinomial": {
        "alpha": [0.1, 0.5, 1.0, 2.0],  # Laplace smoothing
    },
    # Gaussian Mixture Model
    # Fashion MNIST: apenas diagonal (full não converge na prática)
    "gmm": {
        "n_components": [1, 2, 3, 4],  # Componentes Gaussianas por classe
        "covariance_type": ["diag"],  # Apenas diagonal (full não converge)
        "n_init": [10],  # Reinicializações do EM (mais inicializações)
        "max_iter": [500],  # Mais iterações para convergência
        "tol": [1e-4],  # Tolerância menos estrita
    },
    # Regressão Logística Softmax (Multinomial)
    "logistic_softmax": {
        "C": [0.01, 0.1, 1.0, 10.0],  # Regularização L2
        "solver": ["lbfgs"],  # Otimizador
        "max_iter": [2000],  # Iterações
    },
    # Regressão Logística One-vs-Rest
    "logistic_ovr": {
        "C": [0.01, 0.1, 1.0, 10.0],  # Regularização L2
        "solver": ["lbfgs"],  # Otimizador
        "max_iter": [2000],  # Iterações
    },
    # Random Forest
    "random_forest": {
        "n_estimators": [50, 100, 200],  # Número de árvores
        "max_depth": [10, 20, None],  # Profundidade máxima
        "min_samples_split": [2, 5],  # Mínimo para split
        "max_features": ["sqrt", "log2"],  # Features por split
    },
}

# Melhores hiperparâmetros default (baseline)
DEFAULT_PARAMS = {
    "naive_bayes": {
        "var_smoothing": 1e-9,
    },
    "naive_bayes_bernoulli": {
        "alpha": 1.0,
        "binarize": 0.0,
    },
    "naive_bayes_multinomial": {
        "alpha": 1.0,
    },
    "gmm": {
        "n_components": 2,
        "covariance_type": "full",  # Default full para Fashion MNIST
        "n_init": 10,
        "max_iter": 100,
    },
    "logistic_softmax": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "logistic_ovr": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "max_features": "sqrt",
    },
}


def get_param_grid(model_name: str) -> dict:
    """
    Retorna o grid de hiperparâmetros para um modelo no Fashion MNIST

    Args:
        model_name: Nome do modelo

    Returns:
        Dicionário com grid de hiperparâmetros
    """
    if model_name not in HYPERPARAMETER_GRIDS:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(HYPERPARAMETER_GRIDS.keys())}"
        )
    return HYPERPARAMETER_GRIDS[model_name]


def get_default_params(model_name: str) -> dict:
    """
    Retorna os hiperparâmetros default para um modelo no Fashion MNIST

    Args:
        model_name: Nome do modelo

    Returns:
        Dicionário com hiperparâmetros default
    """
    if model_name not in DEFAULT_PARAMS:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(DEFAULT_PARAMS.keys())}"
        )
    return DEFAULT_PARAMS[model_name]


def get_total_combinations(model_name: str) -> int:
    """
    Calcula total de combinações de hiperparâmetros para um modelo

    Args:
        model_name: Nome do modelo

    Returns:
        Número total de combinações
    """
    grid = get_param_grid(model_name)
    total = 1
    for param_values in grid.values():
        total *= len(param_values)
    return total


if __name__ == "__main__":
    # Mostrar grids e combinações
    print("=" * 70)
    print("FASHION MNIST - HYPERPARAMETER GRIDS")
    print("=" * 70)

    for model_name in HYPERPARAMETER_GRIDS.keys():
        grid = get_param_grid(model_name)
        total = get_total_combinations(model_name)

        print(f"\n{model_name.upper()}")
        print("-" * 70)
        print(f"Total combinations: {total}")
        print(f"Grid:")
        for param, values in grid.items():
            print(f"  {param}: {values}")

    print("\n" + "=" * 70)
    print("Estimated total trials (5-fold CV):")
    print("=" * 70)
    total_all = 0
    for model_name in HYPERPARAMETER_GRIDS.keys():
        total = get_total_combinations(model_name)
        trials = total * 5  # 5-fold CV
        total_all += trials
        print(f"  {model_name}: {total} configs × 5 folds = {trials} fits")

    print(f"\nTotal across all models: {total_all} fits")
