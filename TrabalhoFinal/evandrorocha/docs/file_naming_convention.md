# Convenção de Nomenclatura de Arquivos por Modelo

## 📁 Estrutura de Arquivos Atualizada

Agora **todos os arquivos** são salvos com o prefixo do nome do modelo para evitar confusão.

### Para ResNet50

```
models/
├── resnet50_best.pth              # Melhor modelo (checkpoint completo)
├── resnet50_history.json          # Histórico de treinamento
└── resnet50_test_metrics.json     # Métricas no conjunto de teste

runs/
└── resnet50_20251208-183000/      # Logs do TensorBoard
```

### Para EfficientNet-B0

```
models/
├── efficientnet_b0_best.pth
├── efficientnet_b0_history.json
└── efficientnet_b0_test_metrics.json

runs/
└── efficientnet_b0_20251208-184500/
```

### Para DenseNet121

```
models/
├── densenet121_best.pth
├── densenet121_history.json
└── densenet121_test_metrics.json

runs/
└── densenet121_20251208-190000/
```

## 📊 Comparação Entre Modelos

Com essa estrutura, fica fácil comparar modelos:

```bash
# Ver histórico de cada modelo
cat models/resnet50_history.json
cat models/efficientnet_b0_history.json
cat models/densenet121_history.json

# Comparar métricas de teste
cat models/resnet50_test_metrics.json
cat models/efficientnet_b0_test_metrics.json
cat models/densenet121_test_metrics.json
```

## 🔧 Mudanças Implementadas

### 1. Histórico de Treinamento
**Antes:** `history.json` (sobrescrito por cada modelo)  
**Depois:** `{model_name}_history.json`

```python
history_path = os.path.join(self.save_dir, f'{self.model_name}_history.json')
```

### 2. TensorBoard Logs
**Antes:** `runs/20251208-183000/`  
**Depois:** `runs/{model_name}_20251208-183000/`

```python
log_dir = f'./runs/{self.model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
```

### 3. Métricas de Teste
**Antes:** `test_metrics.json` (sobrescrito por cada modelo)  
**Depois:** `{model_name}_test_metrics.json`

```python
test_metrics_path = os.path.join(args.save_dir, f'{args.model}_test_metrics.json')
```

### 4. Checkpoints do Modelo
**Já estava correto:** `{model_name}_best.pth`

```python
path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
```

## 🚀 Uso

### Treinar Diferentes Modelos

```bash
# ResNet50
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model resnet50 --epochs 50 --batch-size 16 --freeze-backbone

# EfficientNet-B0
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model efficientnet_b0 --epochs 50 --batch-size 16 --freeze-backbone

# DenseNet121
docker compose exec tuberculosis-detection-gpu python src/train.py \
  --model densenet121 --epochs 50 --batch-size 16 --freeze-backbone
```

### Visualizar TensorBoard

```bash
# Ver logs de um modelo específico
tensorboard --logdir=runs/resnet50_20251208-183000

# Comparar todos os modelos
tensorboard --logdir=runs/
```

## 📝 Benefícios

✅ **Sem sobrescrita**: Cada modelo tem seus próprios arquivos  
✅ **Fácil comparação**: Nomes claros indicam qual modelo  
✅ **Organização**: Estrutura consistente para todos os modelos  
✅ **Rastreabilidade**: Histórico completo de cada experimento  

## ⚠️ Arquivos Antigos

Você pode ter arquivos sem prefixo de treinamentos anteriores:

```
models/
├── best_model.pth          # ← Antigo (sem prefixo)
├── history.json            # ← Antigo (sem prefixo)
└── test_metrics.json       # ← Antigo (sem prefixo)
```

**Recomendação**: Renomear ou remover para evitar confusão:

```bash
# Dentro do container
docker compose exec tuberculosis-detection-gpu bash
mv models/best_model.pth models/OLD_best_model.pth
mv models/history.json models/OLD_history.json
mv models/test_metrics.json models/OLD_test_metrics.json
```
