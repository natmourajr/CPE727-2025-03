"""
Exemplo de uso do ExperimentTracker com MLflow

Este exemplo demonstra como usar o ExperimentTracker para treinar
e comparar diferentes modelos no Fashion MNIST.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression

from src.data_loader import FashionMNISTLoader
from src.experiment_tracker import ExperimentTracker
from src.utils import set_seed

# Fixar seed para reprodutibilidade
set_seed(42)

print("=" * 80)
print("Exemplo de Experiment Tracking com MLflow")
print("=" * 80)

# 1. Carregar dados
print("\n1. Carregando Fashion MNIST...")
loader = FashionMNISTLoader(flatten=True, normalize=True)
X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

print(f"   Train: {X_train.shape}")
print(f"   Val:   {X_val.shape}")
print(f"   Test:  {X_test.shape}")

# 2. Criar tracker
print("\n2. Inicializando ExperimentTracker...")
tracker = ExperimentTracker(experiment_name="fashion-mnist-example")

# 3. Treinar Naive Bayes
print("\n3. Treinando Naive Bayes Gaussiano...")
nb_model = GaussianNB()
nb_run_id = tracker.track_training(
    model_name="naive_bayes",
    model=nb_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_params={"var_smoothing": 1e-9},
    val_split=(X_val, y_val),
)
print(f"   Run ID: {nb_run_id}")

# 4. Treinar Regressão Logística (OvR)
print("\n4. Treinando Regressão Logística (One-vs-Rest)...")
lr_ovr_model = LogisticRegression(
    multi_class="ovr", max_iter=1000, C=1.0, solver="lbfgs", random_state=42
)
lr_ovr_run_id = tracker.track_training(
    model_name="logistic_ovr",
    model=lr_ovr_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_params={
        "multi_class": "ovr",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    val_split=(X_val, y_val),
)
print(f"   Run ID: {lr_ovr_run_id}")

# 5. Treinar Regressão Logística (Softmax)
print("\n5. Treinando Regressão Logística (Softmax/Multinomial)...")
lr_softmax_model = LogisticRegression(
    multi_class="multinomial", max_iter=1000, C=1.0, solver="lbfgs", random_state=42
)
lr_softmax_run_id = tracker.track_training(
    model_name="logistic_softmax",
    model=lr_softmax_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_params={
        "multi_class": "multinomial",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    val_split=(X_val, y_val),
)
print(f"   Run ID: {lr_softmax_run_id}")

# 6. Comparar resultados
print("\n6. Comparando modelos...")
print("=" * 80)

comparison = tracker.compare_runs(metric="test_accuracy")

if comparison is not None and not comparison.empty:
    print("\nTop 3 modelos por acurácia (teste):")
    print("-" * 80)

    cols_to_show = [
        "tags.model_type",
        "metrics.test_accuracy",
        "metrics.test_f1_macro",
        "metrics.train_time",
    ]

    available_cols = [col for col in cols_to_show if col in comparison.columns]

    if available_cols:
        top_models = comparison[available_cols].head(3)
        print(top_models.to_string(index=False))
    else:
        print("Colunas esperadas não encontradas na comparação")

# 7. Melhor modelo
print("\n7. Melhor modelo:")
print("-" * 80)
best_run = tracker.get_best_run(metric="test_accuracy")

if best_run:
    model_type = best_run.get("tags.model_type", "N/A")
    test_acc = best_run.get("metrics.test_accuracy", 0)
    test_f1 = best_run.get("metrics.test_f1_macro", 0)
    train_time = best_run.get("metrics.train_time", 0)

    print(f"   Modelo: {model_type}")
    print(f"   Acurácia (teste): {test_acc:.4f}")
    print(f"   F1-Score (macro): {test_f1:.4f}")
    print(f"   Tempo de treino: {train_time:.2f}s")

print("\n" + "=" * 80)
print("✓ Experimentos concluídos!")
print("\nPara visualizar os resultados na UI do MLflow, execute:")
print("  mlflow ui")
print("  Acesse: http://localhost:5000")
print("=" * 80)
