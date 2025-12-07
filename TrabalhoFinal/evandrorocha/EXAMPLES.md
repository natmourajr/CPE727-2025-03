# Exemplos de Uso

## 📋 Exemplos Práticos

### 1. Treinar Modelos Individuais

```bash
# Treinar ResNet-50 (padrão)
python src/main.py train

# Treinar DenseNet-121
python src/main.py train --model densenet121

# Treinar EfficientNet-B0 com hiperparâmetros customizados
python src/main.py train --model efficientnet_b0 --batch-size 32 --lr 5e-5 --epochs 100

# Treinar com configurações específicas
python src/main.py train \
    --model resnet101 \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --dropout 0.6 \
    --patience 15
```

### 2. Avaliar Modelos

```bash
# Avaliar modelos treinados
python src/main.py evaluate

# Avaliar modelos específicos
python src/main.py evaluate --models resnet50 densenet121

# Avaliar com configurações customizadas
python src/main.py evaluate \
    --models resnet50 resnet101 densenet121 \
    --batch-size 32 \
    --results-dir ./results/experiment1
```

### 3. Uso com Docker

```bash
# Iniciar container
docker-compose up -d

# Treinar dentro do container
docker-compose exec tuberculosis-detection python src/main.py train --model resnet50

# Avaliar dentro do container
docker-compose exec tuberculosis-detection python src/main.py evaluate

# Ver logs em tempo real
docker-compose logs -f

# Parar container
docker-compose down
```

### 4. Preparar Dataset

```bash
# Organizar dataset baixado
python src/prepare_data.py \
    --source /path/to/downloaded/shenzhen \
    --target ./data/shenzhen

# Apenas verificar dataset existente
python src/prepare_data.py --verify-only --target ./data/shenzhen
```

### 5. Scripts de Conveniência

```bash
# Iniciar ambiente completo
./start.sh

# Treinar todos os modelos sequencialmente
./train_all.sh
```

## 🔬 Experimentos Avançados

### Experimento 1: Comparar Arquiteturas

```bash
# Treinar múltiplas arquiteturas
for model in resnet50 densenet121 efficientnet_b0; do
    python src/main.py train --model $model --epochs 50
done

# Comparar resultados
python src/main.py evaluate --models resnet50 densenet121 efficientnet_b0
```

### Experimento 2: Grid Search de Hiperparâmetros

```bash
# Testar diferentes learning rates
for lr in 1e-3 5e-4 1e-4 5e-5; do
    python src/main.py train \
        --model resnet50 \
        --lr $lr \
        --save-dir ./models/lr_$lr
done
```

### Experimento 3: Diferentes Tamanhos de Imagem

```bash
# Testar diferentes resoluções
for size in 224 299 384; do
    python src/main.py train \
        --model efficientnet_b0 \
        --img-size $size \
        --save-dir ./models/size_$size
done
```

### Experimento 4: Transfer Learning vs From Scratch

```bash
# Com transfer learning (padrão)
python src/main.py train --model resnet50 --pretrained

# Sem transfer learning
python src/main.py train --model resnet50 --save-dir ./models/no_pretrain
```

## 📊 Visualização de Resultados

### Jupyter Notebook

```python
# No notebook
import sys
sys.path.append('../src')

from evaluate import ModelEvaluator
from dataset import create_dataloaders

# Carregar dados
_, _, test_loader = create_dataloaders('./data/shenzhen')

# Criar avaliador
evaluator = ModelEvaluator()

# Carregar e avaliar modelo
model = evaluator.load_model('./models/best_model.pth', 'resnet50')
results = evaluator.evaluate_model(model, test_loader, 'ResNet50')

# Plotar resultados
evaluator.plot_confusion_matrix('ResNet50')
evaluator.plot_roc_curves()
```

### TensorBoard

```bash
# Iniciar TensorBoard
tensorboard --logdir=./runs --port=6006

# Acessar em: http://localhost:6006
```

## 🐛 Debug e Desenvolvimento

### Testar com Subset dos Dados

```python
# Modificar em src/train.py ou criar script de teste
from torch.utils.data import Subset

# Usar apenas 100 amostras para teste rápido
train_subset = Subset(train_dataset, range(100))
train_loader = DataLoader(train_subset, batch_size=16)
```

### Verificar GPU

```bash
# Dentro do container
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Profile de Performance

```python
# Adicionar profiling
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Seu código de treinamento
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 📈 Análise de Resultados

### Carregar Histórico de Treinamento

```python
import json
import matplotlib.pyplot as plt

# Carregar histórico
with open('./models/history.json', 'r') as f:
    history = json.load(f)

# Plotar
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.show()
```

### Comparar Múltiplos Experimentos

```python
import pandas as pd

# Criar DataFrame de comparação
results = {
    'ResNet50': {'acc': 0.95, 'f1': 0.94, 'auc': 0.98},
    'DenseNet121': {'acc': 0.96, 'f1': 0.95, 'auc': 0.99},
    'EfficientNet': {'acc': 0.97, 'f1': 0.96, 'auc': 0.99}
}

df = pd.DataFrame(results).T
print(df)
```

## 🚀 Produção

### Exportar Modelo para ONNX

```python
import torch.onnx

# Carregar modelo
model = load_model('./models/best_model.pth')
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Exportar
torch.onnx.export(
    model,
    dummy_input,
    './models/model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

### Criar API de Inferência

```python
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms

app = Flask(__name__)
model = load_model('./models/best_model.pth')
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open(request.files['image']).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
    
    return jsonify({
        'prediction': 'Tuberculosis' if pred == 1 else 'Normal',
        'confidence': float(prob[0][pred])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```
