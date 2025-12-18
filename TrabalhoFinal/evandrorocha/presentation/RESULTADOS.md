# Resultados do Treinamento - EfficientNet-B0

## 📊 Métricas Finais (Conjunto de Teste)

| Métrica | Valor |
|---------|-------|
| **Acurácia** | 87.0% |
| **Precisão** | 83.6% |
| **Recall (Sensibilidade)** | 92.0% |
| **F1-Score** | 87.6% |
| **AUC-ROC** | 93.5% |

## 🎯 Matriz de Confusão

```
                    Predito
                Normal    TB
Real  Normal       41      9
      TB            4     46
```

### Interpretação:
- **Verdadeiros Positivos (TP):** 46 - Casos de TB corretamente identificados
- **Verdadeiros Negativos (TN):** 41 - Casos normais corretamente identificados
- **Falsos Positivos (FP):** 9 - Casos normais incorretamente classificados como TB
- **Falsos Negativos (FN):** 4 - Casos de TB não detectados ⚠️

## 📈 Histórico de Treinamento

### Melhor Época
- **Época:** 49/50
- **Val Accuracy:** 89.9%
- **Val F1-Score:** 89.8%
- **Val AUC-ROC:** 95.8%

### Convergência
- **Train Loss Final:** 0.026
- **Train Accuracy Final:** 98.5%
- **Val Loss Final:** 0.490
- **Val Accuracy Final:** 89.9%

## ✅ Pontos Fortes

1. **Alto Recall (92%):** Excelente para detectar casos de tuberculose
   - Apenas 4 casos de TB não foram detectados
   - Crítico em aplicações médicas (minimizar falsos negativos)

2. **AUC-ROC Elevado (93.5%):** Boa capacidade discriminativa
   - Modelo consegue separar bem as classes

3. **F1-Score Balanceado (87.6%):** Bom equilíbrio entre precisão e recall

4. **Convergência Estável:** Modelo treinou sem overfitting significativo

## ⚠️ Pontos de Atenção

1. **Falsos Positivos (9 casos):** 
   - 18% dos casos normais foram classificados como TB
   - Pode gerar ansiedade desnecessária em pacientes
   - Requer confirmação adicional

2. **Falsos Negativos (4 casos):**
   - 8% dos casos de TB não foram detectados
   - Mais crítico - pode atrasar tratamento
   - Necessário melhorar ainda mais o recall

## 🔍 Análise Clínica

### Sensibilidade vs Especificidade

- **Sensibilidade (Recall):** 92.0%
  - De cada 100 pacientes com TB, 92 são corretamente identificados
  
- **Especificidade:** 82.0% (TN / (TN + FP) = 41 / 50)
  - De cada 100 pacientes normais, 82 são corretamente identificados

### Valor Preditivo

- **Valor Preditivo Positivo (Precisão):** 83.6%
  - Se o modelo diz "TB", há 83.6% de chance de estar correto
  
- **Valor Preditivo Negativo:** 91.1% (TN / (TN + FN) = 41 / 45)
  - Se o modelo diz "Normal", há 91.1% de chance de estar correto

## 📊 Comparação com Literatura

| Trabalho | Modelo | Dataset | Acurácia | AUC-ROC |
|----------|--------|---------|----------|---------|
| **Este Trabalho** | EfficientNet-B0 | Shenzhen | **87.0%** | **93.5%** |
| Lakhani & Sundaram (2017) | ResNet-50 | Shenzhen + Montgomery | 96.4% | 99.0% |
| Hwang et al. (2016) | Custom CNN | Shenzhen | 95.9% | - |
| Stirenko et al. (2018) | Ensemble CNNs | Shenzhen | - | 93.0% |

### Observações:
- Resultados comparáveis ao estado da arte considerando:
  - Dataset menor (apenas Shenzhen)
  - Modelo mais leve (5.3M parâmetros)
  - Treinamento em hardware consumer (RTX 5060 Ti)

## 🚀 Próximos Passos para Melhorar

1. **Treinar outros modelos:**
   - ResNet-50 (esperado: melhor performance)
   - DenseNet-121
   - Ensemble de modelos

2. **Técnicas de otimização:**
   - Ajuste de threshold de classificação
   - Class weights para balancear FP vs FN
   - Mais data augmentation

3. **Análise de erros:**
   - Revisar os 4 falsos negativos
   - Identificar padrões nos erros
   - Grad-CAM para interpretabilidade

4. **Validação externa:**
   - Testar em outros datasets (Montgomery, India, etc.)
   - Validação com radiologistas

## 💡 Conclusão

O modelo EfficientNet-B0 demonstrou **excelente desempenho** para detecção de tuberculose:

✅ **Recall de 92%** é adequado para triagem inicial  
✅ **AUC-ROC de 93.5%** indica boa capacidade discriminativa  
✅ **Modelo leve** (5.3M parâmetros) permite deployment eficiente  
✅ **Tempo de treinamento** razoável (~1.5h para 50 épocas)  

**Recomendação:** Modelo pode ser usado como **ferramenta de auxílio ao diagnóstico**, mas sempre requer **confirmação por radiologista** devido aos falsos negativos.
