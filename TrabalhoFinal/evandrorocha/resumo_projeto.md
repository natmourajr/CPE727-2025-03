# Resumo do Projeto: Detecção de Tuberculose com Deep Learning

## 1. Objetivo
Desenvolver e comparar modelos de Inteligência Artificial para detecção automática de Tuberculose (TB) em radiografias de tórax, utilizando o dataset Shenzhen China (CXR). O foco principal foi avaliar o impacto do **Transfer Learning** comparando uma CNN treinada do zero (Baseline) contra arquiteturas estado-da-arte pré-treinadas.

## 2. Metodologia
- **Dataset:** 662 imagens de Raio-X (336 Tuberculose, 326 Normais).
- **Split:** Treino (70%), Validação (15%) e Teste (15% = 100 imagens balanceadas).
- **Modelos Avaliados:**
    1.  **SimpleCNN (Baseline):** 4 blocos convolucionais, treinada do zero (1.2M params).
    2.  **ResNet-50:** Transfer Learning (ImageNet), deep residual learning (25.6M params).
    3.  **DenseNet-121:** Feature reuse, conexões densas (8.0M params).
    4.  **EfficientNet-B0:** Compound scaling, eficiência (5.3M params).
- **Regularização:** Data Augmentation, Dropout, Batch Normalization, Weight Decay e Early Stopping.

## 3. Resultados Principais (Conjunto de Teste)

| Modelo | Sensibilidade (Recall) | Especificidade | AUC-ROC | Parâmetros | Conclusão |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ResNet-50** | **90%** | 92% | **96.04%** | 25.6M | **Melhor Performance Clínica** |
| **EfficientNet-B0** | 80% | **94%** | 89.48% | 5.3M | **Melhor Custo-Benefício** (Leve) |
| **DenseNet-121** | 82% | 88% | 86.32% | 8M | **Convergência Mais Rápida** (13 épocas) |
| **CNN Baseline** | 80% | 90% | 90.64% | **1.2M** | **Prova de Conceito** (Ineficiente) |

## 4. Conclusões Chave
1.  **Transfer Learning é Superior:** A ResNet-50 superou significativamente a Baseline e outros modelos, atingindo **96.04% de AUC-ROC**.
2.  **Eficiência vs Performance:** A EfficientNet-B0 provou ser extremamente eficiente, mantendo alta especificidade (94%) com apenas 20% dos parâmetros da ResNet.
3.  **Baseline Surpreendente:** A CNN Baseline, mesmo treinada do zero, atingiu resultados respeitáveis (90% AUC), mas sofreu mais risco de overfitting e precisou de mais épocas.
4.  **Recomendação Clínica:** A **ResNet-50** é o modelo recomendado devido ao melhor equilíbrio entre Sensibilidade (não perder doentes) e Especificidade (não alarmar saudáveis).

## 5. Visualizações Geradas

- Matrizes de Confusão comparativas.
- Gráficos de Sensibilidade vs Especificidade para cada modelo.
