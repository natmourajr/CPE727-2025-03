# 🎯 Guia Rápido - Docker Compose Unificado

## ✅ **PROBLEMA RESOLVIDO!**

Agora você tem **um único `docker-compose.yml`** que funciona em:
- ✅ Mac Apple Silicon (M1/M2/M3)
- ✅ Mac Intel
- ✅ Linux com GPU NVIDIA
- ✅ Linux/Windows sem GPU (CPU)

---

## 🚀 Como Usar no seu Mac M1

### Método 1: Automático (Recomendado)

```bash
# Download do dataset
./download_dataset.sh

# Iniciar ambiente (detecção automática)
./start.sh

# Acesse: http://localhost:8888
```

### Método 2: Manual (especificando profile)

```bash
# Build
COMPOSE_PROFILES=m1 docker-compose build

# Iniciar
COMPOSE_PROFILES=m1 docker-compose up -d

# Acessar
# http://localhost:8888
```

---

## 📋 Profiles Disponíveis

| Profile | Quando Usar | Comando |
|---------|-------------|---------|
| `m1` | Mac Apple Silicon (M1/M2/M3) | `COMPOSE_PROFILES=m1 docker-compose up` |
| `gpu` | Linux/Windows com NVIDIA GPU | `COMPOSE_PROFILES=gpu docker-compose up` |
| `cpu` | Intel/AMD sem GPU | `COMPOSE_PROFILES=cpu docker-compose up` |

---

## 🔧 Comandos Principais

### Mac Apple Silicon (M1/M2/M3)

```bash
# Iniciar (detecção automática)
./start.sh

# OU especificar profile
COMPOSE_PROFILES=m1 docker-compose up -d

# Baixar dataset
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

### Linux/Windows com GPU NVIDIA

```bash
# Iniciar
COMPOSE_PROFILES=gpu docker-compose up -d

# Verificar GPU
COMPOSE_PROFILES=gpu docker-compose exec tuberculosis-detection-gpu nvidia-smi

# Treinar modelo
COMPOSE_PROFILES=gpu docker-compose exec tuberculosis-detection-gpu \
    python src/train.py --model resnet50 --batch-size 16

# Parar
docker-compose down
```

### Intel/AMD CPU apenas

```bash
# Iniciar
COMPOSE_PROFILES=cpu docker-compose up -d

# Treinar modelo
COMPOSE_PROFILES=cpu docker-compose exec tuberculosis-detection-cpu \
    python src/train.py --model resnet50 --batch-size 8

# Parar
docker-compose down
```

---

## 📊 Estrutura do docker-compose.yml

```yaml
services:
  tuberculosis-detection-m1:      # Para Mac M1/M2/M3
    profiles: ["m1", "arm64", "apple-silicon"]
    dockerfile: Dockerfile.m1
    platform: linux/arm64
    
  tuberculosis-detection-gpu:     # Para GPU NVIDIA
    profiles: ["gpu", "intel", "nvidia"]
    dockerfile: Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
    
  tuberculosis-detection-cpu:     # Para CPU apenas
    profiles: ["cpu", "intel-cpu"]
    dockerfile: Dockerfile
```

---

## 🎮 Exemplos Práticos

### 1. Setup Completo (Mac M1)

```bash
# Passo 1: Download dataset
./download_dataset.sh

# Passo 2: Iniciar ambiente
./start.sh

# Passo 3: Treinar modelo
docker-compose exec tuberculosis-detection-m1 \
    python src/train.py --model resnet50 --epochs 50 --batch-size 8

# Passo 4: Avaliar
docker-compose exec tuberculosis-detection-m1 \
    python src/evaluate.py --model resnet50
```

### 2. Treinar Múltiplos Modelos (GPU)

```bash
# Iniciar
COMPOSE_PROFILES=gpu docker-compose up -d

# Treinar ResNet-50
docker-compose exec tuberculosis-detection-gpu \
    python src/train.py --model resnet50 --batch-size 32

# Treinar DenseNet-121
docker-compose exec tuberculosis-detection-gpu \
    python src/train.py --model densenet121 --batch-size 32

# Treinar EfficientNet-B0
docker-compose exec tuberculosis-detection-gpu \
    python src/train.py --model efficientnet_b0 --batch-size 32
```

### 3. Desenvolvimento Iterativo

```bash
# Iniciar Jupyter
./start.sh

# Acessar: http://localhost:8888
# Editar código em ./src/
# Mudanças refletem imediatamente (volume montado)

# Testar no container
docker-compose exec tuberculosis-detection-m1 \
    python src/train.py --model resnet50 --epochs 1 --batch-size 8
```

---


## 🐛 Troubleshooting

### Erro: "failed to register layer" (Mac M1)

✅ **Resolvido!** Agora usa `Dockerfile.m1` automaticamente.

```bash
# Se persistir, rebuild:
COMPOSE_PROFILES=m1 docker-compose build --no-cache
```

### Erro: "no matching manifest for linux/arm64"

```bash
# Certifique-se de usar profile correto
COMPOSE_PROFILES=m1 docker-compose up -d
```

### Container não inicia

```bash
# Ver logs detalhados
docker-compose logs

# Verificar portas
lsof -i :8888

# Rebuild
docker-compose down
COMPOSE_PROFILES=m1 docker-compose build --no-cache
```

### GPU não detectada (Linux)

```bash
# Verificar drivers
nvidia-smi

# Verificar Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Se falhar, use profile CPU
COMPOSE_PROFILES=cpu docker-compose up -d
```

---

## 💡 Dicas Importantes

### Para Mac M1/M2/M3

1. **Batch Size**: Use valores menores (8) para evitar falta de memória
2. **MPS**: Apple Silicon usa Metal Performance Shaders (mais rápido que CPU puro)
3. **Tempo**: Espere ~10-12 min por época com ResNet-50
4. **Desenvolvimento**: Ideal para testes e desenvolvimento
5. **Produção**: Para treinar modelos finais, considere usar GPU CUDA

### Para GPU NVIDIA

1. **Batch Size**: Pode usar valores maiores (16-32)
2. **Monitoramento**: Use `nvidia-smi` para monitorar uso
3. **Memory**: Se der OOM (Out of Memory), reduza batch_size
4. **Performance**: ~5-10x mais rápido que Mac M1

### Para CPU

1. **Paciência**: Treinamento muito mais lento
2. **Batch Size**: Use valores pequenos (4-8)
3. **Epochs**: Considere reduzir número de épocas
4. **Modelo**: Prefira modelos mais leves (ResNet-50 em vez de ResNet-101)

---

## 📚 Arquivos Criados

```
.
├── docker-compose.yml           # ✨ Unificado com profiles
├── Dockerfile                   # Para Intel/AMD (x86_64)
├── Dockerfile.m1                # ✨ Para Apple Silicon (ARM64)
├── start.sh                     # ✨ Detecção automática
├── download_dataset.sh          # ✨ Download com detecção
├── train_all.sh                 # Treinar todos os modelos
├── USAGE.md                     # ✨ Guia de profiles
├── QUICKSTART.md                # Guia rápido
└── README.md                    # Documentação principal
```

---

## 🎯 Próximos Passos

```bash
# 1. Download dataset
./download_dataset.sh

# 2. Iniciar ambiente
./start.sh

# 3. Explorar dados no Jupyter
# http://localhost:8888

# 4. Treinar primeiro modelo
docker-compose exec tuberculosis-detection-m1 \
    python src/train.py --model resnet50 --epochs 10 --batch-size 8

# 5. Avaliar resultados
docker-compose exec tuberculosis-detection-m1 \
    python src/evaluate.py
```

---

## 🔗 Links Úteis

- **Jupyter Lab**: http://localhost:8888
- **Dataset**: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets
- **Docker Compose Profiles**: https://docs.docker.com/compose/profiles/

---

## ✅ Checklist Final

- [x] `docker-compose.yml` unificado criado
- [x] `Dockerfile.m1` para Apple Silicon criado
- [x] `start.sh` com detecção automática criado
- [x] `download_dataset.sh` atualizado
- [x] Scripts tornados executáveis
- [x] Documentação atualizada

**Tudo pronto!** Agora é só executar `./start.sh` no seu Mac M1! 🎉
