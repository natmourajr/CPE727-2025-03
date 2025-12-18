# Guia de Uso - Features Manuais para MLP

Este guia explica como usar features extraídas manualmente (handcrafted features) para treinar o MLP.

## 📊 Features Implementadas

O extrator cria **81 features** divididas em 7 categorias:

### 1. **Intensidade (10 features)**
- Média, desvio padrão, variância
- Mínimo, máximo, mediana
- Quartis (25%, 75%)
- Assimetria (skew) e curtose

### 2. **Histograma (16 features)**
- Distribuição de intensidades em 16 bins

### 3. **GLCM - Textura (20 features)**
- Contraste, dissimilaridade, homogeneidade
- Energia, correlação
- ASM (Angular Second Moment)
- Entropia
- Calculado em 4 direções (0°, 45°, 90°, 135°)

### 4. **LBP - Textura Local (10 features)**
- Local Binary Patterns
- Captura padrões de textura em escala local

### 5. **Momentos de Hu (7 features)**
- Invariantes a rotação, escala e translação
- Úteis para análise de forma

### 6. **Gradiente (8 features)**
- Magnitude e direção do gradiente (Sobel)
- Laplaciano (segunda derivada)

### 7. **FFT - Frequência (10 features)**
- Análise de frequência espacial
- Energia em baixa, média e alta frequência

## 🚀 Como Usar

### Passo 1: Extrair Features do Dataset

```bash
# Extrair features do conjunto de treino
python data/extract_manual_features.py \
    --data-dir data/train \
    --output-dir data/features \
    --split train \
    --num-workers 4

# Extrair features do conjunto de validação
python data/extract_manual_features.py \
    --data-dir data/val \
    --output-dir data/features \
    --split val \
    --num-workers 4

# Extrair features do conjunto de teste
python data/extract_manual_features.py \
    --data-dir data/test \
    --output-dir data/features \
    --split test \
    --num-workers 4
```

**Estrutura esperada dos dados:**
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

**Arquivos gerados:**
```
data/features/
├── train_features_manual.npy    # Features de treino [N, 81]
├── train_labels_manual.npy      # Labels de treino [N]
├── val_features_manual.npy      # Features de validação
├── val_labels_manual.npy
├── test_features_manual.npy     # Features de teste
├── test_labels_manual.npy
└── class_names.txt              # Nomes das classes
```

### Passo 2: Treinar MLP com Features Manuais

```bash
# MLP Simples (recomendado para começar)
python experiments/train_mlp_manual.py \
    --features-dir data/features \
    --model-type simple \
    --epochs 200 \
    --batch-size 32 \
    --lr 0.001 \
    --dropout 0.3 \
    --normalize

# MLP Profundo (mais camadas)
python experiments/train_mlp_manual.py \
    --features-dir data/features \
    --model-type deep \
    --hidden-sizes 128 64 32 \
    --epochs 200 \
    --batch-size 32 \
    --lr 0.001 \
    --dropout 0.4 \
    --normalize
```

### Passo 3: Avaliar o Modelo

```bash
python experiments/evaluate_mlp.py \
    --checkpoint results/mlp_manual_TIMESTAMP/best_model.pth \
    --data-dir data/test \
    --mode two_stage \
    --features-path data/features \
    --save-dir results/evaluation_manual
```

## 📈 Comparação: Manual vs Deep Learning

| Aspecto | Features Manuais | Features Deep (ResNet50) |
|---------|------------------|--------------------------|
| **Número de features** | 81 | 2048 |
| **Tempo de extração** | ~0.1s/imagem | ~0.01s/imagem (GPU) |
| **Interpretabilidade** | ✅ Alta | ❌ Baixa |
| **Performance esperada** | 85-90% AUC | 93-96% AUC |
| **Requer GPU** | ❌ Não | ✅ Sim (recomendado) |
| **Tamanho do modelo** | Pequeno (~50KB) | Grande (~100MB) |

## 💡 Quando Usar Features Manuais?

### ✅ Vantagens
- **Interpretabilidade**: Você sabe exatamente o que cada feature representa
- **Menor complexidade**: Menos parâmetros, treina mais rápido
- **Sem GPU necessária**: Pode rodar em qualquer máquina
- **Análise de features**: Pode identificar quais features são mais importantes
- **Dataset pequeno**: Funciona melhor com poucos dados

### ❌ Desvantagens
- **Performance inferior**: Geralmente 5-10% menor AUC que deep learning
- **Engenharia manual**: Requer conhecimento do domínio
- **Menos flexível**: Features fixas, não aprendem automaticamente

## 🔬 Análise de Features

Após treinar, você pode analisar quais features são mais importantes:

```python
import numpy as np
from data.feature_extraction import ManualFeatureExtractor

# Carrega modelo treinado
# ... (código de carregamento)

# Obtém nomes das features
extractor = ManualFeatureExtractor()
feature_names = extractor.get_feature_names()

# Analisa importância (exemplo com pesos da primeira camada)
weights = model.fc1.weight.data.cpu().numpy()
importance = np.abs(weights).mean(axis=0)

# Top 10 features mais importantes
top_indices = np.argsort(importance)[-10:]
for idx in top_indices[::-1]:
    print(f"{feature_names[idx]}: {importance[idx]:.4f}")
```

## 🎯 Dicas de Otimização

1. **Normalização**: Sempre use `--normalize` para padronizar as features
2. **Dropout**: Comece com 0.3, aumente se houver overfitting
3. **Learning Rate**: 0.001 é um bom ponto de partida
4. **Épocas**: 200 épocas geralmente é suficiente
5. **Batch Size**: 32 funciona bem para a maioria dos casos

## 📊 Resultados Esperados

Com o dataset Shenzhen (~566 imagens):

| Métrica | Valor Esperado |
|---------|---------------|
| Acurácia | 85-90% |
| AUC-ROC | 0.87-0.92 |
| Sensibilidade | 82-88% |
| Especificidade | 88-92% |

## 🔧 Testando o Extrator

Para testar se o extrator está funcionando:

```bash
python data/feature_extraction.py
```

Isso deve imprimir informações sobre as 81 features extraídas de uma imagem de teste.

## 📚 Referências

As features implementadas são baseadas em:
- **GLCM**: Haralick et al. (1973) - Textural Features for Image Classification
- **LBP**: Ojala et al. (2002) - Multiresolution Gray-Scale and Rotation Invariant Texture Classification
- **Hu Moments**: Hu (1962) - Visual Pattern Recognition by Moment Invariants
