"""
Configurações globais do projeto
"""
from pathlib import Path

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
MLRUNS_DIR = RESULTS_DIR / "mlruns"

# Fashion MNIST
FASHION_MNIST_DIR = DATA_DIR / "fashion_mnist"
FASHION_MNIST_NUM_CLASSES = 10
FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# AG_NEWS
AG_NEWS_DIR = DATA_DIR / "ag_news"
AG_NEWS_NUM_CLASSES = 4
AG_NEWS_CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
AG_NEWS_MAX_FEATURES = 10000  # TF-IDF vocabulary size

# Legacy compatibility
NUM_CLASSES = FASHION_MNIST_NUM_CLASSES
CLASS_NAMES = FASHION_MNIST_CLASS_NAMES

# Configurações de treinamento
BATCH_SIZE = 64
RANDOM_SEED = 42

# MLflow
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"
MLFLOW_EXPERIMENT_NAME = "fashion-mnist-comparison"

# Datasets disponíveis
AVAILABLE_DATASETS = ["fashion_mnist", "ag_news"]

# Modelos disponíveis
AVAILABLE_MODELS = [
    # Generativos
    "naive_bayes",              # Gaussian Naive Bayes
    "naive_bayes_bernoulli",    # Bernoulli Naive Bayes
    "naive_bayes_multinomial",  # Multinomial Naive Bayes
    "gmm",                      # Gaussian Mixture Model
    # Discriminativos
    "logistic_ovr",             # Logistic Regression One-vs-Rest
    "logistic_softmax",         # Logistic Regression Softmax
    "random_forest",            # Random Forest
]

# PCA configurations (número de componentes)
PCA_CONFIGS = {
    'baseline': None,  # Sem PCA (784 features para Fashion MNIST)
    'pca_2': 2,        # ~46.8% variância
    'pca_3': 3,        # ~52-55% variância (estimado)
    'pca_10': 10,      # ~72.0% variância
    'pca_50': 50,      # ~86.3% variância
    'pca_100': 100,    # ~91.2% variância
}

# Criar diretórios se não existirem
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR, LOGS_DIR, MLRUNS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
