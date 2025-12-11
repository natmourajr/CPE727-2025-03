# Guia de Uso - SimpleCNN_TB (CNN Tradicional Otimizada)

Este guia explica como usar a **SimpleCNN_TB**, uma CNN tradicional otimizada especificamente para detecção de tuberculose.

## 🎯 Por Que SimpleCNN_TB?

A SimpleCNN_TB foi projetada especificamente para o dataset Shenzhen (~566 imagens):

### ✅ Vantagens
- **Apenas ~500K parâmetros** (vs 51M da SimpleCNN ou 25.6M da ResNet50)
- **Menor risco de overfitting** em datasets pequenos
- **Global Average Pooling** reduz drasticamente os parâmetros
- **4 blocos convolucionais** capturam features em diferentes níveis
- **Performance esperada: 86-89% AUC**

### 📊 Comparação

| Modelo | Parâmetros | AUC Esperado | Ideal Para |
|--------|-----------|--------------|------------|
| LeNetStyle | 5.8M | 75-80% | Aprendizado |
| SimpleCNN | 51M | 82-86% | Dataset médio |
| **SimpleCNN_TB** | **~500K** | **86-89%** | **TB (dataset pequeno)** |
| TraditionalCNN | 30M | 85-88% | Dataset grande |
| ResNet50 | 25.6M | 92-95% | Transfer learning |

## 🏗️ Arquitetura

```
Input: [3, 224, 224]
    ↓
Bloco 1: Conv(32) → BatchNorm → ReLU → MaxPool [32, 112, 112]
    ↓ (Detecta bordas e texturas básicas)
Bloco 2: Conv(64) → BatchNorm → ReLU → MaxPool [64, 56, 56]
    ↓ (Detecta padrões de infiltrados pulmonares)
Bloco 3: Conv(128) → BatchNorm → ReLU → MaxPool [128, 28, 28]
    ↓ (Detecta lesões e nódulos)
Bloco 4: Conv(256) → BatchNorm → ReLU → MaxPool [256, 14, 14]
    ↓ (Detecta cavitações e estruturas complexas)
Global Average Pooling: [256, 14, 14] → [256, 1, 1]
    ↓
FC1: 256 → 128 → ReLU → Dropout(0.4)
    ↓
FC2: 128 → 64 → ReLU → Dropout(0.2)
    ↓
FC3: 64 → 2 (Normal/TB)
```

## 🚀 Como Usar

### 1. Instalação

```bash
pip install -r requirements.txt
```

### 2. Organizar Dados

```
data/
├── train/
│   ├── Normal/
│   │   ├── img1.png
│   │   └── ...
│   └── TB/
│       ├── img1.png
│       └── ...
├── val/
│   ├── Normal/
│   └── TB/
└── test/
    ├── Normal/
    └── TB/
```

### 3. Treinar

```bash
# SimpleCNN_TB (Recomendado para TB)
python experiments/train_cnn_traditional.py \
    --data-dir data \
    --model-type simple_tb \
    --epochs 150 \
    --batch-size 16 \
    --lr 0.0001 \
    --dropout 0.4

# SimpleCNN (Versão original)
python experiments/train_cnn_traditional.py \
    --data-dir data \
    --model-type simple \
    --epochs 150 \
    --batch-size 16 \
    --lr 0.0001 \
    --dropout 0.3

# TraditionalCNN (Mais profunda)
python experiments/train_cnn_traditional.py \
    --data-dir data \
    --model-type traditional \
    --epochs 150 \
    --batch-size 16 \
    --lr 0.0001 \
    --dropout 0.5
```

### 4. Avaliar

```bash
python experiments/evaluate_mlp.py \
    --checkpoint results/cnn_simple_tb_*/best_model.pth \
    --data-dir data/test \
    --mode end_to_end \
    --save-dir results/evaluation_cnn
```

## ⚙️ Hiperparâmetros Recomendados

### Para Dataset Shenzhen (~566 imagens)

```python
# Configuração otimizada
epochs = 150              # Suficiente para convergir
batch_size = 16          # Pequeno para dataset pequeno
learning_rate = 0.0001   # Conservador
weight_decay = 1e-4      # Regularização L2
dropout = 0.4            # Previne overfitting
```

### Data Augmentation (ESSENCIAL!)

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # Flip horizontal
    transforms.RandomRotation(15),               # Rotação ±15°
    transforms.ColorJitter(                      # Ajustes de cor
        brightness=0.2,
        contrast=0.2
    ),
    transforms.RandomAffine(                     # Translação
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## 📈 Resultados Esperados

### Métricas

| Métrica | Valor Esperado |
|---------|---------------|
| Acurácia | 85-88% |
| AUC-ROC | 86-89% |
| Sensibilidade | 83-87% |
| Especificidade | 87-90% |

### Arquivos Gerados

```
results/cnn_simple_tb_TIMESTAMP/
├── best_model.pth              # Melhor modelo (maior AUC)
├── last_model.pth              # Último modelo
├── training_history.png        # Gráficos de treinamento
├── history.json                # Histórico de métricas
└── args.json                   # Argumentos usados
```

## 🔬 Diferenças Principais

### SimpleCNN vs SimpleCNN_TB

| Característica | SimpleCNN | SimpleCNN_TB |
|----------------|-----------|--------------|
| **Blocos Conv** | 3 | 4 |
| **Pooling Final** | Flatten | Global Average Pooling |
| **Parâmetros** | ~51M | ~500K (100x menor!) |
| **FC Layers** | 2 | 3 (menores) |
| **Overfitting** | Médio-Alto | Baixo |
| **AUC Esperado** | 82-86% | 86-89% |

### Global Average Pooling

```python
# SimpleCNN (Flatten)
x = x.view(x.size(0), -1)  # [batch, 128*28*28] = [batch, 100352]
x = fc1(x)  # Precisa de 100352 * 512 = 51M parâmetros!

# SimpleCNN_TB (GAP)
x = gap(x)  # [batch, 256, 14, 14] → [batch, 256, 1, 1]
x = x.view(x.size(0), -1)  # [batch, 256]
x = fc1(x)  # Precisa de apenas 256 * 128 = 33K parâmetros!
```

**Vantagens do GAP:**
- ✅ Reduz drasticamente parâmetros
- ✅ Menos overfitting
- ✅ Mais robusto a variações de posição
- ✅ Usado em arquiteturas modernas (ResNet, Inception)

## 💡 Dicas de Otimização

### 1. Se Overfitting (val_loss aumenta)
```bash
# Aumente dropout
--dropout 0.5

# Aumente weight decay
--weight-decay 5e-4

# Mais data augmentation
# (edite train_cnn_traditional.py)
```

### 2. Se Underfitting (train_loss alto)
```bash
# Diminua dropout
--dropout 0.3

# Aumente learning rate
--lr 0.0005

# Treine por mais épocas
--epochs 200
```

### 3. Para Melhor Performance
```bash
# Use ensemble de 3-5 modelos
# Treine com seeds diferentes e faça média das predições
```

## 🎓 Comparação com ResNet50

| Aspecto | SimpleCNN_TB | ResNet50 |
|---------|--------------|----------|
| **Arquitetura** | Sequencial | Skip connections |
| **Profundidade** | 4 camadas conv | 50 camadas |
| **Parâmetros** | ~500K | ~25.6M |
| **Performance** | 86-89% AUC | 92-95% AUC |
| **Treino** | ~2-3 horas | ~4-6 horas |
| **Interpretabilidade** | ⭐⭐⭐⭐ Alta | ⭐⭐ Média |
| **Overfitting** | ⭐⭐ Baixo | ⭐ Muito baixo* |

*Com transfer learning

## 📚 Uso no Código

```python
from models.traditional_cnn import SimpleCNN_TB

# Criar modelo
model = SimpleCNN_TB(
    num_classes=2,
    dropout_rate=0.4
)

# Forward pass
output = model(images)  # [batch, 2]

# Extrair feature maps (para visualização)
features = model.get_feature_maps(images)
# features['block1']: [batch, 32, 112, 112]
# features['block2']: [batch, 64, 56, 56]
# features['block3']: [batch, 128, 28, 28]
# features['block4']: [batch, 256, 14, 14]
```

## 🎯 Recomendação Final

**Para seu trabalho de detecção de TB:**

1. **Treine SimpleCNN_TB** como baseline de CNN tradicional
2. **Treine ResNet50** (já no seu README) para comparação
3. **Compare resultados** e analise trade-offs

Isso mostrará:
- Evolução de CNNs (tradicional → moderna)
- Impacto de skip connections
- Trade-off entre complexidade e performance
