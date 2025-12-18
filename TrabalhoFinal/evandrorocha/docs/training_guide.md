# Guia de Uso: Treinamento de Modelos

## 🎯 Uso Básico (Recomendado)

### Treinar com configurações padrão

```bash
# ResNet50 (backbone congelado por padrão)
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model resnet50 --epochs 50 --batch-size 16

# EfficientNet-B0
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model efficientnet_b0 --epochs 50 --batch-size 16

# DenseNet121
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model densenet121 --epochs 50 --batch-size 16
```

**Por padrão, o backbone é CONGELADO** ✅ (melhor para dataset pequeno como Shenzhen)

## 🔧 Argumentos Disponíveis

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--model` | `resnet50` | Modelo a treinar |
| `--epochs` | `50` | Número de épocas |
| `--batch-size` | `16` | Tamanho do batch |
| `--lr` | `1e-4` | Learning rate |
| `--data-dir` | `./data/shenzhen` | Diretório dos dados |
| `--save-dir` | `./models` | Onde salvar modelos |
| `--no-freeze-backbone` | `False` | Descongelar backbone (não recomendado) |

## 🧊 Freeze vs No-Freeze

### ✅ Padrão: Backbone Congelado (Recomendado)

```bash
# Simplesmente não passe nenhum argumento adicional
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model resnet50 --epochs 50
```

**Resultado:**
```
✓ Backbone CONGELADO - usando feature extraction (padrão)
  Use --no-freeze-backbone para descongelar
Parâmetros totais: 25,000,000
Parâmetros treináveis: 1,000,000 (4.0%)
```

### ⚠️ Backbone Descongelado (Avançado)

```bash
# Apenas se você tiver um dataset GRANDE (>10k imagens)
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model resnet50 --epochs 50 --no-freeze-backbone
```

**Resultado:**
```
⚠️  Backbone DESCONGELADO - fine-tuning completo
    Isso pode causar overfitting em datasets pequenos!
Parâmetros totais: 25,000,000
Parâmetros treináveis: 25,000,000 (100.0%)
```

## 📊 Arquivos Gerados

Após o treinamento, você terá:

```
models/
├── resnet50_best.pth              # Melhor modelo
├── resnet50_history.json          # Histórico de treinamento
└── resnet50_test_metrics.json     # Métricas finais

runs/
└── resnet50_20251208-183000/      # Logs TensorBoard
```

## 🚀 Exemplos Completos

### Exemplo 1: Treinamento Rápido (Teste)

```bash
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model resnet50 \
  --epochs 5 \
  --batch-size 16
```

### Exemplo 2: Treinamento Completo (Produção)

```bash
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model resnet50 \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-3
```

### Exemplo 3: Fine-tuning Completo (Apenas se necessário)

```bash
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model resnet50 \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-5 \
  --no-freeze-backbone
```

## 📝 Notas Importantes

> **Para o dataset Shenzhen (662 imagens):**
> - ✅ **Use o padrão** (backbone congelado)
> - ❌ **NÃO use** `--no-freeze-backbone`
> - 🎯 Isso evita overfitting e melhora generalização

> **Quando usar `--no-freeze-backbone`:**
> - Dataset muito grande (>10k imagens)
> - Imagens muito diferentes do ImageNet
> - Você tem recursos computacionais suficientes
> - Já testou com backbone congelado primeiro
