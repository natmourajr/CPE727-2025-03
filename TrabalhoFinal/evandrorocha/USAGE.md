# 🚀 Guia de Uso - Docker Compose com Profiles

## 📋 Profiles Disponíveis

Este projeto usa **Docker Compose Profiles** para suportar diferentes arquiteturas automaticamente:

| Profile | Sistema | Hardware | Container |
|---------|---------|----------|-----------|
| `m1` | Mac Apple Silicon | M1/M2/M3 (ARM64) | `tuberculosis-detection-m1` |
| `gpu` | Linux/Windows | NVIDIA GPU | `tuberculosis-detection-gpu` |
| `cpu` | Qualquer | CPU apenas | `tuberculosis-detection-cpu` |

## 🎯 Uso Automático (Recomendado)

O script `start.sh` detecta automaticamente seu sistema:

```bash
# Dar permissões (primeira vez)
chmod +x start.sh download_dataset.sh

# Baixar dataset
./download_dataset.sh

# Iniciar ambiente (detecção automática)
./start.sh

# Acessar Jupyter Lab
# http://localhost:8888
```

## 🔧 Uso Manual por Profile

### Mac Apple Silicon (M1/M2/M3)

```bash
# Build
COMPOSE_PROFILES=m1 docker-compose build

# Iniciar
COMPOSE_PROFILES=m1 docker-compose up -d

# Download dataset
COMPOSE_PROFILES=m1 docker-compose run --rm tuberculosis-detection-m1 \
    python src/download_data.py

# Treinar modelo
COMPOSE_PROFILES=m1 docker-compose exec tuberculosis-detection-m1 \
    python src/train.py --model resnet50 --batch-size 8

# Entrar no container
COMPOSE_PROFILES=m1 docker-compose exec tuberculosis-detection-m1 bash

# Ver logs
docker-compose logs -f tuberculosis-detection-m1

# Parar
docker-compose down
```

### Intel/AMD com GPU NVIDIA

```bash
# Build
COMPOSE_PROFILES=gpu docker-compose build

# Iniciar
COMPOSE_PROFILES=gpu docker-compose up -d

# Verificar GPU
COMPOSE_PROFILES=gpu docker-compose exec tuberculosis-detection-gpu nvidia-smi

# Download dataset
COMPOSE_PROFILES=gpu docker-compose run --rm tuberculosis-detection-gpu \
    python src/download_data.py

# Treinar modelo
COMPOSE_PROFILES=gpu docker-compose exec tuberculosis-detection-gpu \
    python src/train.py --model resnet50 --batch-size 16

# Entrar no container
COMPOSE_PROFILES=gpu docker-compose exec tuberculosis-detection-gpu bash

# Ver logs
docker-compose logs -f tuberculosis-detection-gpu

# Parar
docker-compose down
```

### CPU apenas (qualquer sistema)

```bash
# Build
COMPOSE_PROFILES=cpu docker-compose build

# Iniciar
COMPOSE_PROFILES=cpu docker-compose up -d

# Download dataset
COMPOSE_PROFILES=cpu docker-compose run --rm tuberculosis-detection-cpu \
    python src/download_data.py

# Treinar modelo
COMPOSE_PROFILES=cpu docker-compose exec tuberculosis-detection-cpu \
    python src/train.py --model resnet50 --batch-size 8

# Entrar no container
COMPOSE_PROFILES=cpu docker-compose exec tuberculosis-detection-cpu bash

# Ver logs
docker-compose logs -f tuberculosis-detection-cpu

# Parar
docker-compose down
```

## 📊 Exemplos de Treinamento

### Mac Apple Silicon

```bash
# ResNet-50 (otimizado para M1)
COMPOSE_PROFILES=m1 docker-compose exec tuberculosis-detection-m1 \
    python src/train.py \
    --model resnet50 \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4

# DenseNet-121 (mais leve)
COMPOSE_PROFILES=m1 docker-compose exec tuberculosis-detection-m1 \
    python src/train.py \
    --model densenet121 \
    --epochs 50 \
    --batch-size 8
```

### GPU NVIDIA (Alta Performance)

```bash
# ResNet-50 (batch maior)
COMPOSE_PROFILES=gpu docker-compose exec tuberculosis-detection-gpu \
    python src/train.py \
    --model resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4

# EfficientNet-B4 (modelo pesado)
COMPOSE_PROFILES=gpu docker-compose exec tuberculosis-detection-gpu \
    python src/train.py \
    --model efficientnet_b4 \
    --epochs 50 \
    --batch-size 16
```

### CPU (Economizar recursos)

```bash
# Modelo leve, batch pequeno
COMPOSE_PROFILES=cpu docker-compose exec tuberculosis-detection-cpu \
    python src/train.py \
    --model resnet50 \
    --epochs 30 \
    --batch-size 4 \
    --lr 1e-4
```

## 🔍 Comandos de Diagnóstico

### Verificar qual profile está rodando

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Ver uso de recursos

```bash
# Mac/Linux
docker stats

# GPU (apenas NVIDIA)
COMPOSE_PROFILES=gpu docker-compose exec tuberculosis-detection-gpu nvidia-smi
```

### Verificar dataset

```bash
# Substituir <container-name> pelo container ativo
docker-compose exec <container-name> ls -lh data/shenzhen/
docker-compose exec <container-name> python src/download_data.py --verify-only
```

## 🛠️ Manutenção

### Reconstruir imagem (sem cache)

```bash
# Mac M1
COMPOSE_PROFILES=m1 docker-compose build --no-cache

# GPU
COMPOSE_PROFILES=gpu docker-compose build --no-cache

# CPU
COMPOSE_PROFILES=cpu docker-compose build --no-cache
```

### Limpar containers e volumes

```bash
# Parar e remover containers
docker-compose down

# Remover volumes também (CUIDADO: apaga dados!)
docker-compose down -v

# Limpar sistema Docker
docker system prune -a
```

### Atualizar código sem rebuild

```bash
# O código em ./src é montado como volume
# Apenas edite os arquivos e reinicie o container

docker-compose restart
```

## 📈 Performance Esperada

| Sistema | Batch Size | Tempo/Época (ResNet-50) |
|---------|------------|-------------------------|
| Mac M1 | 8 | ~10-12 min |
| Mac M2 | 8 | ~8-10 min |
| Intel CPU (i7) | 8 | ~15-18 min |
| NVIDIA RTX 3060 | 16 | ~4-5 min |
| NVIDIA RTX 3080 | 32 | ~2-3 min |
| NVIDIA A100 | 32 | ~1-2 min |

## 🐛 Troubleshooting

### Container não inicia

```bash
# Ver erros
docker-compose logs

# Verificar portas
lsof -i :8888  # Mac/Linux
netstat -ano | findstr :8888  # Windows
```

### GPU não detectada (Linux)

```bash
# Verificar drivers
nvidia-smi

# Verificar Docker GPU runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Mac M1: erro "no matching manifest"

```bash
# Certifique-se de usar o profile correto
COMPOSE_PROFILES=m1 docker-compose build --no-cache
```

### Dataset não encontrado

```bash
# Redownload
rm -rf data/shenzhen
./download_dataset.sh
```

## 💡 Dicas

1. **Mac M1**: Use batch_size menor (8) para evitar falta de memória
2. **GPU**: Monitore uso com `nvidia-smi` durante treinamento
3. **CPU**: Considere reduzir epochs ou usar modelos mais leves
4. **Jupyter**: Sempre disponível em `http://localhost:8888`
5. **Volumes**: Código em `./src` é montado, edições refletem instantaneamente

## 🔗 Links Úteis

- **Jupyter Lab**: http://localhost:8888
- **TensorBoard**: `docker-compose exec <container> tensorboard --logdir results/`
- **Dataset**: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html

## 📚 Documentação Adicional

- `README.md` - Visão geral do projeto
- `QUICKSTART.md` - Início rápido
- `EXAMPLES.md` - Exemplos detalhados
