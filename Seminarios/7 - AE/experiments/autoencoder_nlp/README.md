# Autoencoder NLP

Sistema de autoencoder para compressão de características de texto usando o dataset Reuters RCV1.

## Instalação

Este projeto usa `uv` para gerenciamento de dependências. Para instalar as dependências:

```bash
uv sync
```

## Uso

### 1. Download do Dataset

Primeiro, faça o download do dataset Reuters RCV1 e converta-o para formato HDF5:

```bash
uv run autoencoder_nlp download-dataset <diretorio_saida>
```

#### Opções:
- `--n-features, -n`: Número de features mais frequentes a manter (padrão: todas as features)

#### Exemplos:

```bash
# Download com todas as features
uv run autoencoder_nlp download-dataset ./data

# Download com apenas as 2000 features mais frequentes
uv run autoencoder_nlp download-dataset ./data --n-features 2000
```

### 2. Treinamento do Modelo

Para treinar o autoencoder:

```bash
uv run autoencoder_nlp train <caminho_dataset> <diretorio_saida> <dimensoes>
```

#### Argumentos Obrigatórios:
- `caminho_dataset`: Caminho para o arquivo HDF5 do dataset
- `diretorio_saida`: Diretório onde os checkpoints e métricas serão salvos
- `dimensoes`: Lista de dimensões das camadas separadas por vírgula (ex: `2000,500,125,2`)
  - A primeira dimensão deve corresponder ao número de features do dataset
  - A última dimensão é a dimensão do espaço latente
  - As dimensões intermediárias formam a arquitetura do encoder (e são espelhadas no decoder)

#### Opções:

**Modelo:**
- `--activation, -a`: Função de ativação para camadas intermediárias (padrão: `relu`)
  - Opções: `relu`, `tanh`, `sigmoid`, `leaky_relu`
- `--latent-activation`: Função de ativação para o espaço latente (padrão: `linear`)
  - Opções: `linear`, `relu`, `tanh`, `sigmoid`
- `--l1-alpha`: Coeficiente de regularização L1 para ativações latentes (padrão: `None`)

**Otimização:**
- `--learning-rate, --lr`: Taxa de aprendizado para o otimizador AdamW (padrão: `3e-4`)
- `--weight-decay, --wd`: Weight decay para o otimizador AdamW (padrão: `0.01`)

**Treinamento:**
- `--batch-size, -b`: Tamanho do batch (padrão: `32`)
- `--epochs, -e`: Número de épocas de treinamento (padrão: `100`)
- `--val-split`: Fração dos dados para validação (padrão: `0.2`)
- `--num-workers`: Número de workers do DataLoader (use `0` para HDF5) (padrão: `0`)
- `--seed`: Seed aleatória para reprodutibilidade (padrão: `42`)
- `--precision`: Precisão de treinamento (padrão: `32-true`)
  - Opções: `32-true` (float32), `bf16-mixed` (bfloat16 mixed), `16-mixed` (float16 mixed)
  - Use `bf16-mixed` para GPUs modernas (ex: RTX 4070) para melhor desempenho

**Logging:**
- `--mlflow-uri`: URI ou diretório de tracking do MLflow (padrão: `./mlruns`)

#### Exemplos:

```bash
# Treinamento básico
uv run autoencoder_nlp train data/rcv1_dataset.h5 experiments/run_001 2000,500,125,2

# Treinamento com regularização L1
uv run autoencoder_nlp train \
    data/rcv1_dataset.h5 \
    experiments/run_001 \
    2000,500,125,2 \
    --l1-alpha 0.001

# Treinamento com parâmetros personalizados
uv run autoencoder_nlp train \
    data/rcv1_dataset.h5 \
    experiments/run_002 \
    2000,500,125,2 \
    --activation relu \
    --latent-activation linear \
    --l1-alpha 0.001 \
    --learning-rate 3e-4 \
    --weight-decay 0.01 \
    --batch-size 64 \
    --epochs 200 \
    --val-split 0.2 \
    --seed 42

# Treinamento com espaço latente maior
uv run autoencoder_nlp train \
    data/rcv1_dataset.h5 \
    experiments/run_003 \
    2000,500,125,50 \
    --epochs 150 \
    --batch-size 128

# Treinamento para visualização 2D (sem regularização L1)
uv run autoencoder_nlp train \
    data/rcv1_dataset.h5 \
    experiments/viz_2d \
    2000,500,125,2 \
    --epochs 200 \
    --batch-size 128
```

### 3. Avaliação e Codificação para Espaço Latente

Após o treinamento, você pode usar o modelo para codificar o dataset inteiro para o espaço latente:

```bash
uv run autoencoder_nlp evaluate <caminho_dataset> <caminho_checkpoint> <arquivo_saida>
```

#### Argumentos Obrigatórios:
- `caminho_dataset`: Caminho para o arquivo HDF5 do dataset
- `caminho_checkpoint`: Caminho para o checkpoint do modelo treinado (`.ckpt`)
- `arquivo_saida`: Caminho onde as representações latentes serão salvas (`.h5`)

#### Opções:
- `--batch-size, -b`: Tamanho do batch para codificação (padrão: `256`)
  - Batches maiores são mais rápidos mas usam mais memória
- `--device, -d`: Dispositivo a usar (`cuda` ou `cpu`, padrão: auto-detectar)
- `--with-reconstruction, -r`: Também computar erros de reconstrução (padrão: `False`)

#### Exemplos:

```bash
# Codificação básica
uv run autoencoder_nlp evaluate \
    data_2k/rcv1_dataset.h5 \
    experiments/run_001/checkpoints/autoencoder-epoch=50-val_loss=0.1234.ckpt \
    experiments/run_001/latent_representations.h5

# Com análise de erro de reconstrução
uv run autoencoder_nlp evaluate \
    data_2k/rcv1_dataset.h5 \
    experiments/run_001/checkpoints/best.ckpt \
    experiments/run_001/latent_with_errors.h5 \
    --with-reconstruction

# Usando GPU com batch maior
uv run autoencoder_nlp evaluate \
    data_2k/rcv1_dataset.h5 \
    experiments/run_001/checkpoints/best.ckpt \
    experiments/run_001/latent.h5 \
    --batch-size 512 \
    --device cuda
```

#### Arquivo de Saída

O arquivo HDF5 de saída contém:
- `latent_representations`: Array de shape `(num_samples, latent_dim)` com as representações latentes
- `targets`: Listas de índices de labels para cada amostra
- `target_names`: Nomes das categorias
- `reconstruction_errors` (opcional): Erro MSE de reconstrução por amostra (se `--with-reconstruction` foi usado)
- Metadados: configuração do modelo, caminhos, estatísticas

O script também exibe estatísticas úteis sobre o espaço latente (média, desvio padrão, esparsidade, etc.).

## Dicas de Configuração

### Regularização L1 para Visualização 2D

Para criar visualizações 2D do espaço latente (scatterplots):

- **Recomendado**: Use `--l1-alpha None` (sem regularização L1)
  - Com apenas 2 dimensões, L1 pesada pode colapsar uma dimensão a zero
  - O bottleneck de 2D já fornece forte regularização
  - Permite que ambas as dimensões expressem a variância dos dados

- **Alternativa**: Regularização L1 muito leve (`1e-6` ou `1e-5`)
  - Pode criar clusters mais limpos
  - Use apenas se notar visualizações ruidosas/espalhadas

- **Para dimensões maiores** (32, 64, 128): L1 de `1e-5` a `1e-3` é útil para promover esparsidade

### Funções de Ativação

- **ReLU** (padrão): Rápido, cria representações esparsas naturalmente
- **Leaky ReLU**: Recomendado para redes profundas, previne neurônios "mortos"
- **Tanh**: Bom para redes rasas, saídas limitadas [-1, 1]
- **Linear** (espaço latente): Melhor para representações contínuas e visualização

Para dados de texto (TF-IDF), experimente **Leaky ReLU** se o treinamento estagnar.

## Saídas do Treinamento e Avaliação

Após o treinamento, o diretório de saída conterá:

```
experiments/run_001/
├── checkpoints/
│   ├── autoencoder-epoch=XX-val_loss=Y.YYYY.ckpt  # Top 3 melhores modelos
│   └── last.ckpt                                   # Último checkpoint
├── config.yaml                                     # Configuração completa do experimento
├── training_metrics.h5                             # Métricas de treinamento
└── latent_representations.h5                       # Representações latentes (após evaluate)
```

Além disso, os logs do MLflow serão salvos em `./mlruns/` (ou no diretório especificado via `--mlflow-uri`).

## Configuração do Experimento

O arquivo `config.yaml` contém todas as informações do experimento:
- Timestamp
- Informações do dataset
- Arquitetura do modelo
- Hiperparâmetros de treinamento
- Informações do ambiente (versões de bibliotecas, CUDA, etc.)

Este arquivo garante rastreabilidade completa de cada experimento.

## Visualização com MLflow

Para visualizar os experimentos e comparar resultados:

```bash
uv run mlflow ui
```

Acesse `http://localhost:5000` no navegador.

## Eficiência de Memória

O projeto foi otimizado para trabalhar com grandes datasets:

- **Carregamento sob demanda**: As features são convertidas de formato esparso para denso e normalizadas (L2) **sob demanda** durante o treinamento, evitando carregar todo o dataset na memória
- **Formato esparso**: O dataset RCV1 é mantido em formato esparso (~460 MB) ao invés de denso (~6.4 GB para 2000 features)
- **Normalização eficiente**: A normalização L2 é aplicada por amostra durante o `__getitem__`, eliminando picos de memória

Isso permite treinar com datasets grandes mesmo em máquinas com memória limitada.

## Execução com Docker

Além do uso de melhores práticas de empacotamento com uv, o projeto disponibiliza um arquivo Dockerfile para construção de uma imagem de container reprodutível. Para construir a imagem execute:

```shell
docker build -t autoencoder_nlp:latest .
```

Com a imagem construída, pode-se então executar as etapas via Docker, simplesmente adicionaldo `docker run --rm --gpus all autoencoders_nlp:latest` na frente dos comandos. Atente, no entanto, para os diretórios onde serão salvos os arquivos. Um exemplo completo de execução via docker com os dados salvos em ./data ficaria como abaixo:

```shell
docker run -it --rm --gpus all -v ./data:/data autoencoder_nlp \
    uv run autoencoder_nlp download-dataset /data/rcv1_2k --n-features 2000
```

## Ajuda

Para mais informações sobre qualquer comando:

```bash
# Ajuda geral
uv run autoencoder_nlp --help

# Ajuda para download
uv run autoencoder_nlp download-dataset --help

# Ajuda para treinamento
uv run autoencoder_nlp train --help

# Ajuda para avaliação
uv run autoencoder_nlp evaluate --help
```
