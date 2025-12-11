# Guia de Uso - MLP para Detecção de Tuberculose

Este guia explica como usar a implementação do MLP para detecção de tuberculose em radiografias de tórax.

## 📋 Estrutura de Arquivos

```
evandrorocha/
├── models/
│   └── mlp.py                    # Arquiteturas MLP
├── experiments/
│   ├── train_mlp.py             # Script de treinamento
│   └── evaluate_mlp.py          # Script de avaliação
└── data/
    ├── train/                    # Dados de treino
    │   ├── Normal/
    │   └── TB/
    ├── val/                      # Dados de validação
    │   ├── Normal/
    │   └── TB/
    └── test/                     # Dados de teste
        ├── Normal/
        └── TB/
```

## 🚀 Como Usar

### 1. Preparar o Ambiente

```bash
# Instalar dependências
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm seaborn
```

### 2. Organizar os Dados

Organize suas imagens na estrutura acima. Cada pasta (train/val/test) deve conter subpastas com os nomes das classes (Normal e TB).

### 3. Treinar o Modelo

#### Opção A: Modo Two-Stage (Recomendado)

Este modo primeiro extrai features usando ResNet50, depois treina apenas o MLP. É mais rápido e eficiente.

```bash
python experiments/train_mlp.py \
    --data-dir data \
    --mode two_stage \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --hidden-sizes 512 256 128 \
    --dropout 0.5 \
    --save-dir results
```

#### Opção B: Modo End-to-End

Este modo treina o feature extractor e o MLP juntos.

```bash
python experiments/train_mlp.py \
    --data-dir data \
    --mode end_to_end \
    --freeze-extractor \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0001 \
    --hidden-sizes 512 256 128 \
    --dropout 0.5 \
    --save-dir results
```

### 4. Avaliar o Modelo

```bash
# Para modelo two-stage
python experiments/evaluate_mlp.py \
    --checkpoint results/mlp_two_stage_TIMESTAMP/best_model.pth \
    --data-dir data/test \
    --mode two_stage \
    --save-dir results/evaluation

# Para modelo end-to-end
python experiments/evaluate_mlp.py \
    --checkpoint results/mlp_end_to_end_TIMESTAMP/best_model.pth \
    --data-dir data/test \
    --mode end_to_end \
    --save-dir results/evaluation
```

## 📊 Parâmetros Importantes

### Arquitetura do MLP

- **`--hidden-sizes`**: Tamanhos das camadas ocultas
  - Padrão: `512 256 128`
  - Exemplo: `--hidden-sizes 1024 512 256` (MLP maior)
  - Exemplo: `--hidden-sizes 256 128` (MLP menor)

- **`--dropout`**: Taxa de dropout para regularização
  - Padrão: `0.5`
  - Valores típicos: 0.3 a 0.6

### Treinamento

- **`--epochs`**: Número de épocas
  - Padrão: `100`
  - Recomendado: 50-150

- **`--batch-size`**: Tamanho do batch
  - Two-stage: 32-64 (mais rápido)
  - End-to-end: 16-32 (usa mais memória)

- **`--lr`**: Learning rate
  - Two-stage: `0.001` (MLP aprende mais rápido)
  - End-to-end: `0.0001` (mais conservador)

- **`--weight-decay`**: Regularização L2
  - Padrão: `1e-4`

## 📈 Resultados Esperados

Após o treinamento, você encontrará:

```
results/mlp_two_stage_TIMESTAMP/
├── best_model.pth              # Melhor modelo (maior AUC)
├── last_model.pth              # Último modelo
├── training_metrics.png        # Gráficos de treinamento
├── metrics.json                # Histórico de métricas
├── args.json                   # Argumentos usados
├── train_features.npy          # Features de treino (two-stage)
├── train_labels.npy
├── val_features.npy            # Features de validação (two-stage)
└── val_labels.npy
```

Após a avaliação:

```
results/evaluation/
├── evaluation_results.json     # Todas as métricas
├── confusion_matrix.png        # Matriz de confusão
├── roc_curve.png              # Curva ROC
├── test_features.npy          # Features de teste (two-stage)
└── test_labels.npy
```

## 🎯 Métricas Avaliadas

- **Acurácia**: Proporção de predições corretas
- **Precisão**: Proporção de TBs preditos que são realmente TB
- **Recall/Sensibilidade**: Proporção de TBs reais que foram detectados
- **Especificidade**: Proporção de normais reais que foram identificados
- **F1-Score**: Média harmônica entre precisão e recall
- **AUC-ROC**: Área sob a curva ROC (0.5 = aleatório, 1.0 = perfeito)

## 💡 Dicas

1. **Comece com Two-Stage**: É mais rápido e geralmente dá bons resultados
2. **Monitore Overfitting**: Se val_loss aumentar enquanto train_loss diminui, aumente dropout
3. **Ajuste Learning Rate**: Se o modelo não convergir, reduza o learning rate
4. **Experimente Arquiteturas**: Teste diferentes configurações de `--hidden-sizes`
5. **Use Early Stopping**: O modelo salva automaticamente o melhor checkpoint

## 🔬 Número de Variáveis

O MLP usa **2048 features** extraídas do ResNet50:
- Essas features capturam padrões complexos das imagens
- São muito mais eficientes que usar pixels diretamente
- Representam características de alto nível (texturas, formas, etc.)

## ⚙️ Testando a Implementação

Para verificar se tudo está funcionando:

```bash
# Teste os modelos
python models/mlp.py
```

Isso deve imprimir informações sobre cada arquitetura e confirmar que não há erros.

## 📚 Referências

- **Transfer Learning**: Usamos ResNet50 pré-treinado no ImageNet
- **Regularização**: Dropout + Batch Normalization + Weight Decay
- **Otimização**: Adam optimizer com ReduceLROnPlateau scheduler
