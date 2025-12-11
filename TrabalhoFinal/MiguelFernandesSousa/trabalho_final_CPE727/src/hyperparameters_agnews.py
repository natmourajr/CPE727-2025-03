"""
Configuração de hiperparâmetros para AG_NEWS

Define os grids de hiperparâmetros específicos para AG_NEWS.

IMPORTANTE: GMM deve usar APENAS 'diag' covariance para AG_NEWS
(10k features → full covariance = 10k×10k matrix = 100M parameters = memory overflow).
"""

# Grid de hiperparâmetros para cada modelo no AG_NEWS
HYPERPARAMETER_GRIDS = {
    # Naive Bayes Gaussiano
    # Não é ideal para TF-IDF, mas testamos para comparação
    "naive_bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],  # Laplace smoothing
    },
    # Naive Bayes Bernoulli
    # Funciona bem com TF-IDF binarizado
    # IMPORTANTE: Sempre use binarize com TF-IDF (TF-IDF é contínuo, mas BernoulliNB precisa de binário)
    "naive_bayes_bernoulli": {
        "alpha": [0.1, 0.5, 1.0, 2.0],  # Laplace smoothing
        "binarize": [0.0, 0.1],  # 0.0 = binariza >0, 0.1 = threshold mais agressivo
        # Removido None pois TF-IDF contínuo causa problemas numéricos no BernoulliNB
    },
    # Naive Bayes Multinomial
    # Modelo IDEAL para TF-IDF (features de contagem/frequência)
    "naive_bayes_multinomial": {
        "alpha": [0.01, 0.1, 0.5, 1.0],  # Laplace smoothing (valores menores para TF-IDF)
    },
    # Gaussian Mixture Model
    # AG_NEWS: APENAS diag (10k features → full covariance impossível)
    "gmm": {
        "n_components": [1, 2, 3, 4],  # Componentes Gaussianas por classe
        "covariance_type": ["diag"],  # APENAS diag para AG_NEWS
        "n_init": [5, 10],  # Reinicializações do EM
        "max_iter": [100, 200],  # Iterações do EM
    },
    # Regressão Logística Softmax (Multinomial)
    # Excelente para classificação de texto
    "logistic_softmax": {
        "C": [0.1, 1.0, 10.0, 100.0],  # Regularização L2 (maiores para texto)
        "solver": ["lbfgs", "saga"],  # saga suporta L1/L2
        "max_iter": [1000, 2000],  # Iterações
    },
    # Regressão Logística One-vs-Rest
    "logistic_ovr": {
        "C": [0.1, 1.0, 10.0, 100.0],  # Regularização L2
        "solver": ["lbfgs", "saga"],  # saga suporta L1/L2
        "max_iter": [1000, 2000],  # Iterações
    },
    # Random Forest
    # Funciona bem com TF-IDF de alta dimensão
    "random_forest": {
        "n_estimators": [100, 200, 300],  # Mais árvores para texto
        "max_depth": [20, 30, None],  # Profundidade maior para texto
        "min_samples_split": [2, 5],  # Mínimo para split
        "max_features": ["sqrt", "log2", 0.1],  # Features por split
    },
}

# Melhores hiperparâmetros default (baseline)
DEFAULT_PARAMS = {
    "naive_bayes": {
        "var_smoothing": 1e-9,
    },
    "naive_bayes_bernoulli": {
        "alpha": 1.0,
        "binarize": None,  # Sem binarização por default
    },
    "naive_bayes_multinomial": {
        "alpha": 1.0,  # Padrão para TF-IDF
    },
    "gmm": {
        "n_components": 2,
        "covariance_type": "diag",  # APENAS diag para AG_NEWS
        "n_init": 10,
        "max_iter": 100,
    },
    "logistic_softmax": {
        "C": 10.0,  # Regularização menor para texto
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "logistic_ovr": {
        "C": 10.0,  # Regularização menor para texto
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "random_forest": {
        "n_estimators": 200,  # Mais árvores para texto
        "max_depth": None,
        "min_samples_split": 2,
        "max_features": "sqrt",
    },
}


def get_param_grid(model_name: str) -> dict:
    """
    Retorna o grid de hiperparâmetros para um modelo no AG_NEWS

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
    Retorna os hiperparâmetros default para um modelo no AG_NEWS

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


# Aliases para compatibilidade com hyperparameter_tuning_agnews.py
HYPERPARAMETER_GRIDS_AGNEWS = HYPERPARAMETER_GRIDS
CV_FOLDS_AGNEWS = 3  # Default CV folds for AG_NEWS (menor que Fashion MNIST devido ao dataset ser maior)
SCORING_METRICS_AGNEWS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
get_total_combinations_agnews = get_total_combinations


if __name__ == "__main__":
    # Mostrar grids e combinações
    print("=" * 70)
    print("AG_NEWS - HYPERPARAMETER GRIDS")
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

    # Validação crítica
    print("\n" + "=" * 70)
    print("CRITICAL VALIDATIONS:")
    print("=" * 70)
    gmm_cov = get_param_grid("gmm")["covariance_type"]
    if "full" in gmm_cov:
        print("  ⚠️  WARNING: GMM has 'full' covariance - WILL CAUSE MEMORY OVERFLOW!")
        print("  ⚠️  AG_NEWS has 10k features → full covariance = 10k×10k matrix")
    else:
        print("  ✓ GMM using only 'diag' covariance (safe for 10k features)")
