# Comparação de Modelos de Baseline e Aprendizado Profundo para Classificação de Imagens e Notícias

## Trabalho Final - CPE727: Aprendizado Profundo

**Nome:** Miguel Fernandes de Sousa
**Email:** <miguel.sousa@coppe.ufrj.br>
**CRID:** 125074229
**Programa:** PEE/COPPE/UFRJ
**Período:** 2025/3

---

## Resumo

Este trabalho avalia o desempenho de modelos de aprendizado profundo em problemas de classificação multiclasse, utilizando dois datasets distintos: **Fashion MNIST** (imagens, 10 classes, 784 features) e **AG_NEWS** (texto, 4 classes). Foram implementados e avaliados modelos de baseline (Naive Bayes, Gaussian Mixture Models, Regressão Logística e Random Forest) e modelos de aprendizado profundo (CNN para Fashion MNIST e LSTM para AG_NEWS).

Os experimentos incluíram otimização de hiperparâmetros com validação cruzada por grid search em dois estágios, com rastreamento de métricas, parâmetros e artefatos via **MLflow**.

---

## Objetivos

Comparar modelos **generativos** (Naive Bayes, GMM) e **discriminativos** (Regressão Logística, Random Forest) com modelos de **aprendizado profundo** (CNN, LSTM) em problemas de classificação multiclasse, utilizando datasets de modalidades distintas (imagens e texto).

---

## Conjuntos de Dados

### Fashion MNIST (Imagens)

- **Total de amostras:** 70.000 imagens
  - Treino: 48.000 | Validação: 12.000 | Teste: 10.000
- **Dimensionalidade:** 28×28 pixels = 784 features
- **Tipo de dado:** Escala de cinza, normalizado para [-1.0, 1.0], float32
- **Classes:** 10 categorias balanceadas (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

### AG_NEWS (Texto)

- **Total de amostras:** 127.600 notícias
  - Treino: 96.000 | Validação: 24.000 | Teste: 7.600
- **Representação:**
  - **Baseline (TF-IDF):** 10.000 features, com min_df=5 e max_df=0.5
  - **Deep Learning (LSTM):** Tokenização com vocabulário de 10.000 palavras, sequências de comprimento 200
- **Tipo de dado:** Texto (título + descrição curta de notícias)
- **Classes:** 4 categorias balanceadas (World, Sports, Business, Sci/Tech)

---

## Metodologia

### Modelos de Baseline

#### Modelos Generativos
- **Naive Bayes:** Gaussiano, Bernoulli, Multinomial
- **Gaussian Mixture Models (GMM):** 1-4 componentes por classe, covariância diagonal

#### Modelos Discriminativos
- **Regressão Logística:** Softmax e One-vs-Rest (OvR) com regularização L2
- **Random Forest:** 100-300 árvores, otimização de max_depth e max_features

### Modelos de Aprendizado Profundo

#### CNN para Fashion MNIST
**Arquitetura:**
- Conv1: 1→32 filtros 3×3, ReLU, MaxPool 2×2
- Conv2: 32→64 filtros 3×3, ReLU, MaxPool 2×2
- Flatten: 1600 features
- FC1: 1600→128, ReLU, Dropout
- FC2: 128→10 classes

**Hiperparâmetros finais:** learning_rate=0.001, dropout=0.4, batch_size=32, epochs=20

#### LSTM para AG_NEWS
**Arquitetura:**
- Embedding: 10.000 vocab → embedding_dim
- LSTM bidirecional: embedding_dim → hidden_dim
- Dropout
- FC: hidden_dim → 4 classes

**Hiperparâmetros finais:** learning_rate=0.005, embedding_dim=100, hidden_dim=128, dropout=0.4, bidirectional=True, batch_size=32, epochs=20

### Otimização de Hiperparâmetros

**Estratégia de dois estágios:**
1. **Estágio 1 (amostra pequena):** Exploração rápida do espaço de hiperparâmetros
   - Fashion MNIST: 500 amostras, 3 épocas
   - AG_NEWS: 1.000 amostras, 3 épocas
2. **Estágio 2 (amostra maior):** Refinamento dos melhores hiperparâmetros
   - Fashion MNIST: 5.000 amostras, 10 épocas
   - AG_NEWS: 10.000 amostras, 5 épocas
3. **Treinamento final:** Dataset completo com mais épocas (20)

---

## Resultados Principais

### Fashion MNIST

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **CNN** | **92.29%** | **92.30%** | **92.29%** | **92.28%** |
| Logistic OvR | 83.50% | 83.74% | 83.85% | 83.78% |
| Logistic Softmax | 83.40% | 83.85% | 83.75% | 83.76% |
| Naive Bayes Multinomial | 65.55% | 65.42% | 65.55% | 62.76% |
| Naive Bayes Bernoulli | 64.82% | 66.58% | 64.82% | 63.98% |
| Naive Bayes Gaussiano | 59.10% | 62.03% | 58.40% | 55.74% |

**Gap entre discriminativo e generativo:** 17.95 pontos percentuais (Logistic OvR vs. melhor Naive Bayes)

### AG_NEWS

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic OvR | 91.34% | 91.32% | 91.34% | 91.32% |
| Logistic Softmax | 91.24% | 91.22% | 91.24% | 91.22% |
| Random Forest | 90.96% | 90.94% | 90.96% | 90.92% |
| **LSTM** | **89.17%** | **89.30%** | **89.17%** | **89.19%** |
| Naive Bayes Multinomial | 89.66% | 89.61% | 89.66% | 89.62% |
| Naive Bayes Bernoulli | 89.36% | 89.31% | 89.36% | 89.31% |
| GMM | 86.89% | 86.91% | 86.89% | 86.90% |
| Naive Bayes Gaussiano | 86.64% | 86.68% | 86.64% | 86.64% |

**Gap entre discriminativo e generativo:** 1.68 pontos percentuais (Logistic OvR vs. melhor Naive Bayes)

---

## Principais Conclusões

1. **Modalidade dos dados importa:** A diferença de desempenho entre modelos generativos e discriminativos é maior em imagens (17.95 pp) do que em texto (1.68 pp), indicando que features TF-IDF esparsas são mais adequadas para modelagem generativa que pixels correlacionados.

2. **Distribuição no Naive Bayes:** MultinomialNB e BernoulliNB apresentaram desempenho equivalente em ambos datasets, ambos superando GaussianNB em ~6 pp (Fashion MNIST) e ~3 pp (AG_NEWS).

3. **Deep Learning:**
   - **CNN superou todos os baselines** no Fashion MNIST (92.29% vs. 83.50% da melhor Regressão Logística)
   - **LSTM ficou abaixo dos baselines** no AG_NEWS (89.17% vs. 91.34% da Regressão Logística), sugerindo que features TF-IDF permitem separação linear adequada para este dataset

4. **Otimização eficiente:** A estratégia de dois estágios reduziu o tempo de experimentação de 8-12 horas para ~40-60 minutos, mantendo qualidade dos resultados.

---

## Estrutura do Repositório

```
.
├── README.md                           # Este arquivo
├── trabalho_final_CPE727/             # Documento LaTeX do trabalho
├── eda/                               # Análise exploratória dos dados
│   └── outputs/
│       ├── fashion_mnist/
│       └── ag_news/
├── confusion_matrices_baseline/       # Matrizes de confusão (baseline)
├── results/                           # Resultados dos modelos de deep learning
│   └── plots/
└── src/                               # Código-fonte
    └── dataloaders/
```

---

## Tecnologias Utilizadas

- **Python 3.x**
- **PyTorch:** Implementação de CNN e LSTM
- **scikit-learn:** Modelos baseline e validação cruzada
- **MLflow:** Rastreamento de experimentos
- **NumPy, Pandas:** Manipulação de dados
- **Matplotlib, Seaborn:** Visualizações

---

## Bibliografia Principal

1. Ng, A., & Jordan, M. (2002). *On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes*. NIPS 2002.

2. Zheng et al. (2023). *Revisiting Discriminative vs. Generative Classifiers: Theory and Implications*. arXiv:2302.02334.

3. Bouzidi et al. (2024). *Convolutional neural networks and vision transformers for fashion mnist classification: A literature review*. arXiv:2406.03478.

4. Ozdemir, S. (2024). *News Classification with State-of-the-Art Deep Learning Methods*. IDAP 2024.

5. Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.

---

## Autor

**Miguel Fernandes de Sousa**
Programa de Engenharia Elétrica (PEE)
COPPE/UFRJ
Email: miguel.sousa@coppe.ufrj.br
