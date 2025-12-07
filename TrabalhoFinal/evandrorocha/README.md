# Comparação de Modelos de Deep Learning para Detecção de Tuberculose em Radiografias de Tórax

Este repositório contém o código e os experimentos para comparar diferentes arquiteturas de redes neurais profundas na detecção de tuberculose, utilizando o dataset público **Shenzhen Hospital X-ray Set**.

## 🖥️ Compatibilidade Universal

✅ **Funciona em qualquer sistema** usando Docker Compose com profiles automáticos:

| Sistema | Arquitetura | Suporte | Profile |
|---------|-------------|---------|---------|
| 🍎 **Mac M1/M2/M3** | ARM64 | ✅ Aceleração MPS | `m1` |
| 💻 **Mac Intel** | x86_64 | ✅ CPU | `cpu` |
| 🐧 **Linux** | x86_64 | ✅ NVIDIA GPU / CPU | `gpu` / `cpu` |
| 🪟 **Windows** | x86_64 | ✅ NVIDIA GPU / CPU | `gpu` / `cpu` |

**Detecção Automática:** O script `./start.sh` detecta seu sistema e escolhe o profile correto automaticamente!

## 📊 Dataset

O **Shenzhen Hospital X-ray Set** contém radiografias de tórax (CXR) coletadas no Shenzhen No.3 Hospital na China, incluindo:
- **566 imagens** no total
- **326 casos normais**
- **240 casos com manifestações de tuberculose**
- Resolução variável (aproximadamente 3000x3000 pixels)
- Formato: PNG

**Fonte oficial**: [NIH Clinical Center - Tuberculosis Chest X-ray Image Data Sets](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html)

## 🏗️ Estrutura do Projeto

```
.
├── data/                      # Dados do dataset
│   └── shenzhen/
│       ├── normal/           # Imagens normais
│       └── tuberculosis/     # Imagens com TB
├── src/                      # Código fonte
│   ├── download_data.py     # Download automático do dataset
│   ├── dataset.py           # DataLoader customizado
│   ├── models.py            # Arquiteturas de modelos
│   ├── train.py             # Script de treinamento
│   ├── evaluate.py          # Avaliação e comparação
│   └── prepare_data.py      # Preparação dos dados
├── models/                   # Modelos treinados salvos
├── results/                  # Resultados e visualizações
├── notebooks/               # Jupyter notebooks
├── Dockerfile              # Configuração Docker
├── docker-compose.yml      # Orquestração Docker
└── requirements.txt        # Dependências Python
```

## 🚀 Como Usar

### Pré-requisitos

- Docker e Docker Compose instalados
- GPU NVIDIA com drivers CUDA (recomendado)
- Pelo menos 8GB de RAM
- 10GB de espaço em disco

### 1. Baixar o Dataset

Você tem **duas opções** para baixar o dataset de Shenzhen:

#### 🎯 Opção A: Download Automático (Recomendado)

Use o script automatizado que tenta fazer o download e organiza tudo:

```bash
# Método mais simples - script bash completo
./download_dataset.sh

# OU passo a passo:
docker-compose build
docker-compose run --rm tuberculosis-detection python src/download_data.py

# Verificar se o download foi bem-sucedido
docker-compose run --rm tuberculosis-detection python src/download_data.py --verify-only
```

#### 📥 Opção B: Download Manual

Se o download automático falhar (pode ocorrer devido a restrições do site NIH), siga estes passos:

1. **Acesse o site oficial:**
   - URL: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets

2. **Baixe o dataset:**
   - Localize **"Shenzhen Hospital X-ray Set"**
   - Clique em Download para obter `ChinaSet_AllFiles.zip` (~440 MB)

3. **Coloque o arquivo na pasta correta:**
   ```bash
   # Coloque o arquivo baixado aqui:
   data/shenzhen_dataset.zip
   ```

4. **Organize o dataset:**
   ```bash
   # Extrair e organizar automaticamente
   docker-compose run --rm tuberculosis-detection python src/download_data.py
   
   # OU se já extraiu manualmente:
   docker-compose run --rm tuberculosis-detection python src/download_data.py \
       --organize-only \
       --source /caminho/para/ChinaSet_AllFiles
   ```

5. **Verificar:**
   ```bash
   docker-compose run --rm tuberculosis-detection python src/download_data.py --verify-only
   ```

**Estrutura esperada após o download:**
```
data/
└── shenzhen/
    ├── normal/          # 326 imagens de casos normais
    └── tuberculosis/    # 240 imagens com tuberculose
```

### 2. Iniciar o Ambiente Docker

```bash
# Dar permissão de execução aos scripts
chmod +x start.sh train_all.sh

# Construir e iniciar container
./start.sh

# OU manualmente:
docker-compose build
docker-compose up -d
```

Acesse o Jupyter Lab em: `http://localhost:8888`

### 3. Treinar os Modelos

Dentro do container ou usando o script:

```bash
# Treinar um modelo específico
docker-compose exec tuberculosis-detection python src/train.py

# Ou treinar todos os modelos
./train_all.sh
```

### 4. Avaliar e Comparar Modelos

```bash
docker-compose exec tuberculosis-detection python src/evaluate.py
```

## 🎯 Modelos Implementados

Este projeto implementa e compara as seguintes arquiteturas:

1. **ResNet-50**: Rede residual com 50 camadas
2. **ResNet-101**: Versão mais profunda do ResNet
3. **DenseNet-121**: Rede densamente conectada
4. **DenseNet-169**: Versão mais profunda do DenseNet
5. **EfficientNet-B0**: Arquitetura eficiente e escalável
6. **VGG-16**: Arquitetura clássica de CNN

Todos os modelos utilizam:
- **Transfer Learning** com pesos pré-treinados no ImageNet
- **Data Augmentation** para melhorar generalização
- **Early Stopping** para evitar overfitting
- **Learning Rate Scheduling** adaptativo

## 📈 Métricas de Avaliação

Os modelos são avaliados usando:

- **Accuracy**: Acurácia geral
- **Precision**: Precisão na detecção de TB
- **Recall/Sensitivity**: Taxa de verdadeiros positivos
- **F1-Score**: Média harmônica entre precisão e recall
- **AUC-ROC**: Área sob a curva ROC
- **AUC-PR**: Área sob a curva Precision-Recall
- **Confusion Matrix**: Matriz de confusão

## 🛠️ Técnicas Utilizadas

### Data Augmentation
- Rotação aleatória (±15°)
- Flip horizontal
- Ajuste de brilho e contraste
- Shift e scale aleatórios

### Regularização
- Dropout (0.5)
- Weight Decay (L2 regularization)
- Batch Normalization

### Otimização
- Adam optimizer
- Learning rate inicial: 1e-4
- ReduceLROnPlateau scheduler
- Early stopping (patience=10)

## 📊 Resultados Esperados

Os resultados incluem:

- **Curvas de treinamento** (loss e accuracy)
- **Matrizes de confusão** para cada modelo
- **Curvas ROC** comparativas
- **Curvas Precision-Recall** comparativas
- **Tabela comparativa** de métricas
- **Checkpoints** dos melhores modelos

## 🔧 Customização

### Alterar hiperparâmetros

Edite o arquivo `src/train.py`:

```python
BATCH_SIZE = 16          # Tamanho do batch
IMAGE_SIZE = (224, 224)  # Tamanho das imagens
NUM_EPOCHS = 50          # Número de épocas
LEARNING_RATE = 1e-4     # Taxa de aprendizado
```

### Adicionar novos modelos

Edite `src/models.py` e adicione sua arquitetura personalizada.

## 📝 Comandos Úteis Docker

```bash
# Ver logs em tempo real
docker-compose logs -f

# Parar o container
docker-compose down

# Entrar no container
docker-compose exec tuberculosis-detection bash

# Reinstalar dependências
docker-compose exec tuberculosis-detection pip install -r requirements.txt

# Limpar recursos Docker
docker system prune -a
```

## 🐛 Troubleshooting

### Erro de GPU não encontrada

Se você não tem GPU NVIDIA, edite `docker-compose.yml` e remova a seção `deploy`.

### Erro de memória

Reduza o `BATCH_SIZE` em `src/train.py`.

### Dataset não encontrado

Verifique se o dataset está organizado corretamente em `./data/shenzhen/`.

## 📚 Referências

- [Deep Learning for Tuberculosis Detection](https://www.nature.com/articles/s41598-019-42557-4)
- [Transfer Learning for Medical Image Analysis](https://arxiv.org/abs/1902.07208)
- [Shenzhen Hospital X-ray Set](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html)

## 👥 Autor

- Evandro Rocha

