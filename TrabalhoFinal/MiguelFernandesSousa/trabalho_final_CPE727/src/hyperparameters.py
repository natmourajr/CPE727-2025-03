"""
Configuração de hiperparâmetros para Grid Search CV

Define os grids de hiperparâmetros a serem explorados para cada modelo
durante a fase de validação (hyperparameter tuning).
"""

# Grid de hiperparâmetros para cada modelo
HYPERPARAMETER_GRIDS = {
    # Naive Bayes Gaussiano
    # Poucos hiperparâmetros (modelo simples)
    "naive_bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],  # Laplace smoothing
    },
    # Gaussian Mixture Model
    # Hiperparâmetros mais impactantes
    "gmm": {
        "n_components": [1, 2, 3, 4],  # Componentes Gaussianas por classe
        "covariance_type": ["full", "diag"],  # Tipo de covariância
        "n_init": [5, 10],  # Reinicializações do EM
        "max_iter": [100, 200],  # Iterações do EM
    },
    # Regressão Logística Softmax (Multinomial)
    "logistic_softmax": {
        "C": [0.01, 0.1, 1.0, 10.0],  # Regularização L2 (reduzido)
        "solver": ["lbfgs"],  # Otimizador (apenas lbfgs)
        "max_iter": [2000],  # Iterações (apenas 2000)
    },
    # Regressão Logística One-vs-Rest
    "logistic_ovr": {
        "C": [0.01, 0.1, 1.0, 10.0],  # Regularização L2 (reduzido)
        "solver": ["lbfgs"],  # Otimizador (apenas lbfgs)
        "max_iter": [2000],  # Iterações (apenas 2000)
    },
}

# Melhores hiperparâmetros default (baseline para começar)
DEFAULT_PARAMS = {
    "naive_bayes": {
        "var_smoothing": 1e-9,
    },
    "gmm": {
        "n_components": 2,
        "covariance_type": "full",
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
}


def get_param_grid(model_name: str) -> dict:
    """
    Retorna o grid de hiperparâmetros para um modelo

    Args:
        model_name: Nome do modelo ('naive_bayes', 'gmm', 'logistic_softmax', 'logistic_ovr')

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
    Retorna os hiperparâmetros default para um modelo

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
    print("=" * 60)
    print("HYPERPARAMETER GRIDS - Grid Search Configuration")
    print("=" * 60)

    for model_name in HYPERPARAMETER_GRIDS.keys():
        grid = get_param_grid(model_name)
        total = get_total_combinations(model_name)

        print(f"\n{model_name.upper()}")
        print("-" * 60)
        print(f"Total combinations: {total}")
        print(f"Grid:")
        for param, values in grid.items():
            print(f"  {param}: {values}")

    print("\n" + "=" * 60)
    print("Estimated total trials (5-fold CV):")
    print("=" * 60)
    for model_name in HYPERPARAMETER_GRIDS.keys():
        total = get_total_combinations(model_name)
        trials = total * 5  # 5-fold CV
        print(f"  {model_name}: {total} configs × 5 folds = {trials} fits")
