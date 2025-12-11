# Trabalho Final CPE727 - Aprendizado Profundo

Comparação entre modelos generativos e discriminativos para classificação multiclasse em dois datasets: Fashion MNIST (imagens) e AG_NEWS (texto).

**Autor:** Miguel Fernandes de Sousa
**CRID:** 125074229
**Período:** 2025/3

---

## 🚀 Quick Start

### Opção 1: Docker (Mais Simples)

```bash
# Executar todos os experimentos
docker-compose up experiments

# Ver resultados no MLflow
docker-compose up mlflow
# Acesse http://localhost:5000
```

### Opção 2: uv (Desenvolvimento Local)

```bash
# Instalar dependências
uv sync

# Executar experimentos
uv run run_experiments.py
```

---

## Estrutura do Projeto

```
trabalho_final_CPE727/
├── src/                          # Código fonte principal
│   ├── models/                   # Implementações dos modelos
│   │   ├── naive_bayes.py
│   │   ├── naive_bayes_bernoulli.py
│   │   ├── naive_bayes_multinomial.py
│   │   ├── logistic_softmax.py
│   │   ├── logistic_ovr.py
│   │   ├── gmm.py
│   │   ├── random_forest.py
│   │   └── hierarchical_classifier.py
│   ├── preprocessing/            # Transformações de dados
│   ├── data_loader.py            # Carregador Fashion MNIST
│   ├── data_loader_agnews.py     # Carregador AG_NEWS (TF-IDF)
│   ├── data_loader_agnews_tokenized.py  # Carregador AG_NEWS (LSTM)
│   ├── models_deep.py            # Modelos CNN e LSTM
│   ├── train.py                  # Treinamento baseline Fashion MNIST
│   ├── train_agnews.py           # Treinamento baseline AG_NEWS
│   ├── train_deep.py             # Treinamento deep learning
│   ├── hyperparameter_tuning.py  # Tuning Fashion MNIST
│   ├── hyperparameter_tuning_agnews.py  # Tuning AG_NEWS
│   ├── final_evaluation.py       # Avaliação final Fashion MNIST
│   └── final_evaluation_agnews.py  # Avaliação final AG_NEWS
├── scripts/                      # Scripts organizados
│   ├── baseline/                 # Experimentos baseline
│   ├── deep_learning/            # Experimentos deep learning
│   ├── evaluation/               # Scripts de avaliação
│   ├── utilities/                # Utilitários
│   └── eda/                      # Análise exploratória
├── eda/                          # Outputs da análise exploratória
├── mlruns/                       # Experimentos MLflow
├── results/                      # Resultados e gráficos
├── confusion_matrices_baseline/  # Matrizes de confusão baseline
├── v2_deep_apresentacao/         # Apresentação e relatório
└── pyproject.toml                # Dependências do projeto
```

## Datasets

### Fashion MNIST (Imagens)
- **Amostras:** 70.000 imagens (60k treino, 10k teste)
- **Dimensionalidade:** 28x28 pixels = 784 features
- **Classes:** 10 categorias de roupas (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Pré-processamento:** Normalização min-max para [-1, 1] (ou [0, 1] para MultinomialNB)

### AG_NEWS (Texto)
- **Amostras:** 127.600 notícias (120k treino, 7.6k teste)
- **Classes:** 4 categorias (World, Sports, Business, Sci/Tech)
- **Representação:**
  - **Baseline:** TF-IDF com 10.000 features, max_df=0.5, min_df=5
  - **LSTM:** Tokenização word-level, vocabulário 10.000 palavras, sequências de 200 tokens

## Requisitos

### Método 1: Docker (Recomendado)

Requer apenas Docker e Docker Compose instalados:

```bash
# Verificar instalação do Docker
docker --version
docker-compose --version
```

### Método 2: uv (Recomendado para desenvolvimento local)

```bash
# Instalar uv (package manager Python moderno)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sincronizar dependências (cria .venv automaticamente)
uv sync
```

### Método 3: pip/poetry (alternativa)

```bash
# Com pip
pip install torch torchvision scikit-learn numpy pandas matplotlib seaborn mlflow

# Com poetry
poetry install
```

## Como Executar os Experimentos

### Execução com Docker (Método Recomendado)

#### Executar todos os experimentos

```bash
# Construir imagem e rodar todos os experimentos
docker-compose up experiments

# Visualizar resultados no MLflow UI (em outro terminal)
docker-compose up mlflow
# Acesse http://localhost:5000
```

#### Executar experimentos específicos

```bash
# Apenas EDA
docker-compose run eda

# Apenas baseline (Fashion MNIST e AG_NEWS)
docker-compose --profile baseline up

# Apenas deep learning
docker-compose --profile deep up deep-learning

# Experimentos específicos por dataset
docker-compose --profile baseline up baseline-fashion
docker-compose --profile baseline up baseline-agnews
```

#### MLflow UI standalone

```bash
docker-compose up mlflow
# Acesse http://localhost:5000
```

### Execução com uv (Desenvolvimento Local)

Todos os comandos abaixo devem ser executados com `uv run`:

#### Pipeline completo

```bash
# Executar todos os experimentos
uv run run_experiments.py

# Executar apenas Fashion MNIST
uv run run_experiments.py --dataset fashion_mnist

# Executar apenas AG_NEWS
uv run run_experiments.py --dataset ag_news

# Pular EDA
uv run run_experiments.py --skip-eda

# Apenas baseline (sem deep learning)
uv run run_experiments.py --skip-deep

# Apenas deep learning (sem baseline)
uv run run_experiments.py --skip-baseline
```

#### Experimentos individuais

##### 1. Análise Exploratória dos Dados (EDA)

Execute a análise exploratória para entender os datasets:

```bash
# Fashion MNIST
uv run scripts/eda/fashion_mnist_eda.py

# AG_NEWS
uv run scripts/eda/ag_news_eda.py
```

Os outputs serão salvos em `eda/outputs/`.

##### 2. Experimentos Baseline - Fashion MNIST

###### Fase 1: Otimização de Hiperparâmetros

```bash
# Executar Grid Search CV para todos os modelos baseline
uv run src/hyperparameter_tuning.py
```

**Modelos avaliados:**
- Naive Bayes (Gaussiano, Bernoulli, Multinomial)
- Gaussian Mixture Models (GMM)
- Regressão Logística (Softmax e One-vs-Rest)
- Random Forest

**Hiperparâmetros otimizados:**
- **Naive Bayes:** var_smoothing ∈ {1e-09, 1e-08, 1e-07, 1e-06, 1e-05}
- **GMM:** n_components ∈ {1, 2, 3, 4}, covariance_type ∈ {'full', 'diag'}
- **Logistic Regression:** C ∈ {0.01, 0.1, 1.0, 10.0}
- **Random Forest:** n_estimators ∈ {100, 200}, max_depth ∈ {None, 10, 20}, max_features ∈ {'sqrt', 'log2'}

#### Fase 2: Avaliação Final

```bash
# Treinar modelos com melhores hiperparâmetros e avaliar no conjunto de teste
uv run src/final_evaluation.py
```

**Resultados esperados (Test Accuracy):**
- Logistic OvR: 83.50%
- Logistic Softmax: 83.40%
- Naive Bayes Multinomial: 65.55%
- Naive Bayes Bernoulli: 64.82%
- Naive Bayes Gaussiano: 59.10%

### 3. Experimentos Baseline - AG_NEWS

#### Fase 1: Otimização de Hiperparâmetros

```bash
# Executar Grid Search CV para todos os modelos baseline
uv run src/hyperparameter_tuning_agnews.py
```

**Estratégia em 2 fases** (para reduzir consumo de memória):
- **Fase 1:** Quick test com 5k amostras, 2 folds (10 min)
- **Fase 2:** Refinamento com 30k amostras, 2 folds (40 min)

#### Fase 2: Avaliação Final

```bash
# Treinar modelos com melhores hiperparâmetros e avaliar no conjunto de teste
uv run src/final_evaluation_agnews.py
```

**Resultados esperados (Test Accuracy):**
- Logistic OvR: 91.34%
- Logistic Softmax: 91.24%
- Random Forest: 90.96%
- Naive Bayes Multinomial: 89.66%
- Naive Bayes Bernoulli: 89.36%
- GMM: 86.89%
- Naive Bayes Gaussiano: 86.64%

### 4. Experimentos Deep Learning - CNN (Fashion MNIST)

#### Otimização de Hiperparâmetros (2 estágios)

```bash
# Executar otimização em 2 estágios
uv run src/train_deep.py --dataset fashion_mnist --mode grid_search
```

**Estágio 1 (Tiny - 500 amostras):**
- learning_rate ∈ {0.001, 0.01}
- dropout ∈ {0.3, 0.5}
- batch_size ∈ {32, 64}
- epochs = 3

**Estágio 2 (Small - 5000 amostras):**
- learning_rate ∈ {0.0005, 0.001, 0.002}
- dropout ∈ {0.3, 0.4, 0.5}
- batch_size ∈ {32, 64}
- epochs = 10

#### Treinamento Final

```bash
# Treinar modelo final com melhores hiperparâmetros (dataset completo)
uv run scripts/deep_learning/run_lenet_final.py
```

**Arquitetura CNN:**
- Conv1: 1→32 filtros 3x3, ReLU, MaxPool 2x2
- Conv2: 32→64 filtros 3x3, ReLU, MaxPool 2x2
- FC1: 1600→128, ReLU, Dropout
- FC2: 128→10 (saída)

**Hiperparâmetros finais:**
- learning_rate = 0.001
- dropout = 0.4
- batch_size = 32
- epochs = 20

**Resultado esperado:** Accuracy = 92.29%

#### Gerar Matriz de Confusão

```bash
uv run scripts/deep_learning/gen_confusion_fashion_cnn.py
```

### 5. Experimentos Deep Learning - LSTM (AG_NEWS)

#### Otimização de Hiperparâmetros (2 estágios)

```bash
# Executar otimização em 2 estágios
uv run src/train_deep.py --dataset ag_news --mode grid_search
```

**Estágio 1 (Tiny - 1000 amostras):**
- learning_rate ∈ {0.001, 0.01}
- embedding_dim ∈ {50, 100}
- hidden_dim ∈ {64, 128}
- dropout ∈ {0.3, 0.5}
- bidirectional ∈ {False, True}
- batch_size ∈ {32, 64}
- epochs = 3

**Estágio 2 (Small - 10000 amostras):**
- learning_rate ∈ {0.005, 0.01}
- embedding_dim = 100
- hidden_dim ∈ {64, 128}
- dropout ∈ {0.2, 0.3, 0.4}
- bidirectional = True
- batch_size = 32
- epochs = 5

#### Treinamento Final

```bash
# Treinar modelo final com melhores hiperparâmetros (dataset completo)
uv run scripts/deep_learning/train_lstm_final_only.py
```

**Arquitetura LSTM:**
- Embedding: vocab_size (10000) → embedding_dim (100)
- LSTM Bidirecional: embedding_dim → hidden_dim (128)
- Dropout: 0.4
- FC: 2×hidden_dim → 4 classes

**Hiperparâmetros finais:**
- learning_rate = 0.005
- embedding_dim = 100
- hidden_dim = 128
- dropout = 0.4
- bidirectional = True
- batch_size = 32
- epochs = 20

**Resultado esperado:** Accuracy = 89.17%, tempo de treino ≈88.5 minutos

### 6. Executar Todos os Experimentos (Script Unificado)

```bash
# Executar pipeline completo
uv run run_experiments.py
```

Este script executará sequencialmente:
1. EDA para ambos datasets
2. Baseline Fashion MNIST (tuning + avaliação)
3. Baseline AG_NEWS (tuning + avaliação)
4. CNN Fashion MNIST (tuning + treinamento final)
5. LSTM AG_NEWS (tuning + treinamento final)

## Visualização de Resultados

### MLflow UI

Todos os experimentos são rastreados via MLflow:

```bash
mlflow ui
```

Acesse http://localhost:5000 para visualizar:
- Métricas de validação cruzada
- Hiperparâmetros testados
- Modelos salvos
- Artefatos (matrizes de confusão, curvas de aprendizado)

### Análise de Resultados

```bash
# Analisar resultados do MLflow
uv run scripts/evaluation/analyze_mlflow_results.py

# Verificar modelos faltantes
uv run scripts/evaluation/check_missing_models.py
```

## Principais Resultados

### Comparação Baseline vs Deep Learning

| Dataset | Melhor Baseline | Deep Learning | Ganho |
|---------|----------------|---------------|-------|
| Fashion MNIST | 83.50% (Logistic OvR) | 92.29% (CNN) | +8.79 pp |
| AG_NEWS | 91.34% (Logistic OvR) | 89.17% (LSTM) | -2.17 pp |

### Insights Principais

1. **Modelos discriminativos superam generativos** em ambos datasets
2. **Gap maior em imagens** (17.95 pp) vs texto (1.68 pp)
3. **CNN superior para imagens** devido à captura de features hierárquicas
4. **LSTM inferior a baseline em texto** (não usa embeddings pré-treinados como BERT)
5. **Escolha da distribuição importa:** MultinomialNB ≈ BernoulliNB > GaussianNB
6. **Separação linear adequada** para TF-IDF (Logistic Regression > Random Forest)

## Troubleshooting

### Consumo de Memória (AG_NEWS)

Se encontrar problemas de memória com Random Forest:
- Usar estratégia de 2 fases (5k → 30k amostras)
- Desabilitar paralelismo aninhado (n_jobs=1 no modelo dentro do GridSearchCV)
- Reduzir n_jobs do GridSearchCV

### PyTorch

```bash
# Diagnosticar problemas PyTorch
bash scripts/utilities/diagnose_pytorch.sh

# Tentar fix automático
bash scripts/utilities/fix_pytorch.sh
```

## Referências

1. Ng & Jordan (2002). On Discriminative vs. Generative Classifiers
2. Zheng et al. (2023). Revisiting Discriminative vs. Generative Classifiers
3. Bouzidi et al. (2024). CNNs and Vision Transformers for Fashion MNIST
4. Ozdemir (2024). News Classification with Deep Learning Methods

## Relatório Completo

Consulte o relatório completo em LaTeX:
```
v2_deep_apresentacao/relatorio/RELATORIO.tex
```

## Contato

Miguel Fernandes de Sousa
PEE/COPPE/UFRJ
CRID: 125074229
