"""
Modelos de classificacao para Fashion MNIST e AG_NEWS

Implementacoes de classificadores generativos e discriminativos:
    Generativos:
        - Naive Bayes Gaussiano (baseline, assume features contínuas)
        - Naive Bayes Bernoulli (features binárias)
        - Naive Bayes Multinomial (features de contagem, TF-IDF)
        - Gaussian Mixture Model (multimodal, clusters)
    Discriminativos:
        - Regressao Logistica Softmax (multinomial)
        - Regressao Logistica OvR (one-vs-rest)
        - Random Forest (ensemble de árvores)
"""
from src.models.naive_bayes import NaiveBayesGaussian
from src.models.naive_bayes_bernoulli import NaiveBayesBernoulli
from src.models.naive_bayes_multinomial import NaiveBayesMultinomial
from src.models.gmm import GMMClassifier
from src.models.logistic_softmax import LogisticRegressionSoftmax
from src.models.logistic_ovr import LogisticRegressionOvR
from src.models.random_forest import RandomForest

__all__ = [
    "NaiveBayesGaussian",
    "NaiveBayesBernoulli",
    "NaiveBayesMultinomial",
    "GMMClassifier",
    "LogisticRegressionSoftmax",
    "LogisticRegressionOvR",
    "RandomForest",
]
