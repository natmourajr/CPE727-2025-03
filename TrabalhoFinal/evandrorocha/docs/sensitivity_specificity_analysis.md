# Análise de Sensibilidade e Especificidade - ResNet50

## 📊 Resultados do Modelo ResNet50

### Métricas Gerais
- **Accuracy**: 91.00%
- **Precision**: 91.84%
- **Recall (Sensibilidade)**: 90.00%
- **F1-Score**: 90.91%
- **AUC-ROC**: 96.04%

### 🎯 Sensibilidade e Especificidade

#### Sensibilidade (Recall): **90.00%**
- **Definição**: Capacidade de detectar quem TEM tuberculose
- **Cálculo**: TP / (TP + FN) = 45 / (45 + 5) = 90.00%
- **Interpretação**: De 50 pacientes com TB, o modelo detectou corretamente 45
- **✅ EXCELENTE** - Modelo detecta a maioria dos casos de TB

#### Especificidade: **92.00%**
- **Definição**: Capacidade de identificar quem NÃO TEM tuberculose  
- **Cálculo**: TN / (TN + FP) = 46 / (46 + 4) = 92.00%
- **Interpretação**: De 50 pacientes normais, o modelo identificou corretamente 46
- **✅ EXCELENTE** - Poucos falsos positivos

## 📋 Matriz de Confusão

```
                Predito
              Normal   TB
Real Normal     46     4     ← Especificidade: 46/50 = 92%
     TB          5    45     ← Sensibilidade: 45/50 = 90%
```

### Detalhamento

| Métrica | Valor | Significado |
|---------|-------|-------------|
| **True Negatives (TN)** | 46 | Pacientes normais corretamente identificados |
| **False Positives (FP)** | 4 | Pacientes normais diagnosticados como TB (erro) |
| **False Negatives (FN)** | 5 | Pacientes com TB não detectados (erro crítico!) |
| **True Positives (TP)** | 45 | Pacientes com TB corretamente detectados |

## 💡 Interpretação Clínica

### ✅ Pontos Fortes

1. **Alta Sensibilidade (90%)**
   - Detecta 9 em cada 10 casos de tuberculose
   - Importante para triagem e detecção precoce
   - Reduz risco de casos não diagnosticados

2. **Alta Especificidade (92%)**
   - Identifica corretamente 92% dos pacientes saudáveis
   - Poucos falsos alarmes
   - Reduz custos com exames desnecessários

3. **Balanceamento**
   - Sensibilidade e Especificidade bem equilibradas
   - Não sacrifica um em detrimento do outro

### ⚠️ Pontos de Atenção

1. **5 Falsos Negativos**
   - 5 pacientes com TB não foram detectados
   - Em contexto clínico, isso é crítico
   - Esses pacientes precisariam de exames adicionais

2. **4 Falsos Positivos**
   - 4 pacientes saudáveis diagnosticados como TB
   - Causaria ansiedade e exames desnecessários
   - Mas é preferível a não detectar TB real

## 📈 Comparação com Literatura

### Benchmarks para Detecção de TB

| Métrica | Nosso Modelo | Literatura Típica | Status |
|---------|--------------|-------------------|--------|
| Sensibilidade | 90.00% | 85-95% | ✅ Dentro do esperado |
| Especificidade | 92.00% | 80-90% | ✅ Acima da média |
| AUC-ROC | 96.04% | 90-95% | ✅ Excelente |

## 🎯 Recomendações

### Para Uso Clínico

1. **Triagem Inicial**: ✅ Modelo adequado
   - Alta sensibilidade detecta maioria dos casos
   - Pode ser usado como primeira linha de triagem

2. **Diagnóstico Definitivo**: ⚠️ Usar com cautela
   - Sempre confirmar com exames adicionais
   - Não substituir diagnóstico médico especializado

3. **Casos Suspeitos**:
   - Se modelo indica TB → Fazer exames confirmatórios
   - Se modelo indica Normal mas há sintomas → Investigar mais

### Para Melhorar o Modelo

1. **Reduzir Falsos Negativos**:
   - Ajustar threshold de decisão (favorecer sensibilidade)
   - Usar ensemble de modelos
   - Aumentar dataset de casos positivos

2. **Análise de Erros**:
   - Investigar os 5 casos de FN: O que têm em comum?
   - Investigar os 4 casos de FP: Características específicas?

## 📊 Visualização

Gráfico salvo em: `results/resnet50_sensitivity_specificity.png`

O gráfico mostra:
- Comparação visual entre Sensibilidade e Especificidade
- Matriz de confusão detalhada
- Valores percentuais para fácil interpretação

## 🔬 Contexto Médico

### Por que Sensibilidade é Crítica para TB?

- **Doença contagiosa**: Casos não detectados podem infectar outros
- **Tratamento precoce**: Quanto antes detectar, melhor o prognóstico
- **Saúde pública**: Controle epidemiológico depende de detecção

### Por que Especificidade também Importa?

- **Custo**: Exames confirmatórios são caros
- **Ansiedade**: Falsos positivos causam estresse desnecessário
- **Recursos**: Sistema de saúde tem capacidade limitada

## 📝 Conclusão

O modelo ResNet50 apresenta **desempenho excelente** para detecção de tuberculose:

- ✅ Sensibilidade de 90% (detecta maioria dos casos)
- ✅ Especificidade de 92% (poucos falsos alarmes)
- ✅ Balanceamento adequado entre as métricas
- ✅ AUC-ROC de 96% (excelente capacidade discriminativa)

**Adequado para**: Triagem inicial e suporte ao diagnóstico médico  
**Não substitui**: Avaliação clínica e exames confirmatórios especializados
