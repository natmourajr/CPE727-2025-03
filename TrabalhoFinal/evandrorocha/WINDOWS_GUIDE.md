# 🪟 Guia Completo - Windows com GPU NVIDIA

## 📋 Pré-requisitos

### 1. NVIDIA GPU Drivers
Verifique se os drivers estão instalados:
```powershell
nvidia-smi
```
Deve mostrar informações da sua GPU. Se não funcionar, instale de: https://www.nvidia.com/drivers

### 2. Docker Desktop para Windows
- Download: https://www.docker.com/products/docker-desktop
- **Importante**: Habilite integração com **WSL 2** durante a instalação

### 3. NVIDIA Container Toolkit
Siga o guia oficial: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Para instalar no WSL 2:
```bash
# No terminal WSL (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 4. Verificar Configuração
```powershell
# Testar acesso do Docker à GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```
Se mostrar informações da GPU, está tudo OK! ✅

---

## 🚀 Início Rápido (3 comandos)

```batch
REM 1. Download do dataset
download_dataset_windows.bat

REM 2. Iniciar ambiente
start_windows.bat

REM 3. Treinar todos os modelos
train_all_windows.bat
```

---

## 📁 Arquivos para Windows

Este projeto inclui scripts específicos para Windows:

| Arquivo | Descrição |
|---------|-----------|
| `start_windows.bat` | Iniciar ambiente com GPU |
| `download_dataset_windows.bat` | Download do dataset |
| `train_all_windows.bat` | Treinar todos os modelos |

---

## 🔧 Uso Detalhado

### 1. Download do Dataset

```batch
REM Método automático
download_dataset_windows.bat

REM OU comando direto
set COMPOSE_PROFILES=gpu
docker compose run --rm tuberculosis-detection-gpu python src/download_data.py
```

**Download Manual** (se automático falhar):
1. Acesse: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets
2. Baixe "Shenzhen Hospital X-ray Set" (ChinaSet_AllFiles.zip)
3. Coloque em `data\shenzhen_dataset.zip`
4. Execute novamente o script

### 2. Iniciar Ambiente

```batch
REM Usando script
start_windows.bat

REM OU manualmente
set COMPOSE_PROFILES=gpu
docker compose up -d
```

Acesse Jupyter Lab: **http://localhost:8888**

### 3. Verificar GPU

```powershell
# Status da GPU
docker compose exec tuberculosis-detection-gpu nvidia-smi

# Monitorar GPU em tempo real
docker compose exec tuberculosis-detection-gpu nvidia-smi -l 1

# Detalhes de uso
docker compose exec tuberculosis-detection-gpu nvidia-smi dmon -s pucvmet
```

### 4. Treinar Modelos

#### Treinar Todos os Modelos
```batch
train_all_windows.bat
```

#### Treinar Modelo Específico
```powershell
# ResNet-50 com batch size 32
docker compose exec tuberculosis-detection-gpu python src/train.py ^
    --model resnet50 ^
    --epochs 50 ^
    --batch-size 32 ^
    --lr 1e-4

# DenseNet-121
docker compose exec tuberculosis-detection-gpu python src/train.py ^
    --model densenet121 ^
    --epochs 50 ^
    --batch-size 32

# EfficientNet-B0
docker compose exec tuberculosis-detection-gpu python src/train.py ^
    --model efficientnet_b0 ^
    --epochs 50 ^
    --batch-size 16
```

### 5. Avaliar Modelos

```powershell
# Avaliar todos os modelos treinados
docker compose exec tuberculosis-detection-gpu python src/evaluate.py

# Avaliar modelo específico
docker compose exec tuberculosis-detection-gpu python src/evaluate.py --model resnet50
```

### 6. Jupyter Lab

```powershell
# Acessar no navegador
start http://localhost:8888

# Ver token (se necessário)
docker compose logs tuberculosis-detection-gpu | findstr token
```

---

## 📊 Performance no Windows com GPU

### Comparação de Performance

| GPU | Batch Size | Tempo/Época (ResNet-50) |
|-----|------------|-------------------------|
| RTX 3060 | 16 | ~4-5 min |
| RTX 3070 | 32 | ~3-4 min |
| RTX 3080 | 32 | ~2-3 min |
| RTX 3090 | 32 | ~1.5-2 min |
| RTX 4090 | 64 | ~1 min |

### vs Mac M1
- Windows GPU: **5-10x mais rápido**
- Mac M1: ~12 min/época (CPU/MPS)
- Windows GPU: ~2-3 min/época (RTX 3080)

### Configurações Recomendadas

| GPU VRAM | Batch Size | Modelo |
|----------|------------|--------|
| 6GB | 8-16 | ResNet-50, DenseNet-121 |
| 8GB | 16-24 | ResNet-101, DenseNet-169 |
| 10GB+ | 32-64 | EfficientNet-B4, ResNet-152 |

---

## 🎯 Comandos Úteis Windows

### Gerenciamento de Containers

```powershell
# Ver containers rodando
docker compose ps

# Ver logs em tempo real
docker compose logs -f

# Parar container
docker compose down

# Reiniciar container
docker compose restart

# Entrar no container
docker compose exec tuberculosis-detection-gpu bash

# Executar comando único
docker compose exec tuberculosis-detection-gpu python --version
```

### Monitoramento GPU

```powershell
# Status simples
docker compose exec tuberculosis-detection-gpu nvidia-smi

# Monitoramento contínuo (atualiza a cada 1s)
docker compose exec tuberculosis-detection-gpu nvidia-smi -l 1

# Uso de memória
docker compose exec tuberculosis-detection-gpu nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Temperatura
docker compose exec tuberculosis-detection-gpu nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
```

### Manutenção Docker

```powershell
# Reconstruir imagem (sem cache)
docker compose down
set COMPOSE_PROFILES=gpu
docker compose build --no-cache

# Limpar recursos não utilizados
docker system prune -a

# Limpar volumes (CUIDADO: apaga dados!)
docker compose down -v

# Ver espaço em disco usado pelo Docker
docker system df
```

---

## 🐛 Troubleshooting

### 1. GPU não detectada

**Sintoma:** `nvidia-smi` não funciona dentro do container

**Soluções:**
```powershell
# Verificar drivers no host
nvidia-smi

# Verificar NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Reinstalar NVIDIA Container Toolkit (no WSL)
sudo apt-get install --reinstall nvidia-docker2
sudo systemctl restart docker
```

### 2. Erro "docker: unknown flag: --gpus"

**Causa:** Docker Desktop desatualizado

**Solução:**
- Atualizar Docker Desktop para versão mais recente
- Habilitar integração com WSL 2
- Reiniciar Docker Desktop

### 3. Erro de memória (OOM - Out Of Memory)

**Sintoma:** Container trava ou erro "CUDA out of memory"

**Soluções:**
```powershell
# Reduzir batch size
docker compose exec tuberculosis-detection-gpu python src/train.py --batch-size 8

# Usar modelo mais leve
docker compose exec tuberculosis-detection-gpu python src/train.py --model resnet50

# Verificar memória GPU disponível
docker compose exec tuberculosis-detection-gpu nvidia-smi
```

### 4. Porta 8888 ocupada

**Solução:**
```powershell
# Verificar processo usando a porta
netstat -ano | findstr :8888

# Matar processo (substitua <PID> pelo número)
taskkill /PID <PID> /F

# OU mudar porta no docker-compose.yml
# ports: "8889:8888"
```

### 5. Erro de permissão no Windows

**Solução:**
```powershell
# Executar PowerShell como Administrador
# Adicionar usuário ao grupo docker-users
net localgroup docker-users "SEU_USUARIO" /add

# Reiniciar para aplicar mudanças
```

### 6. Dataset não encontrado

**Solução:**
```powershell
# Verificar estrutura
dir data\shenzhen

# Re-download
rmdir /s data\shenzhen
download_dataset_windows.bat

# Verificar
docker compose exec tuberculosis-detection-gpu python src/download_data.py --verify-only
```

---

## 📚 Estrutura do Projeto

```
.
├── docker-compose.yml              # Configuração unificada
├── Dockerfile                      # Para Intel/AMD + GPU NVIDIA
├── Dockerfile.m1                   # Para Mac M1 (não usado no Windows)
├── start_windows.bat              # ⭐ Iniciar ambiente (Windows)
├── download_dataset_windows.bat   # ⭐ Download dataset (Windows)
├── train_all_windows.bat          # ⭐ Treinar modelos (Windows)
├── data/
│   └── shenzhen/                  # Dataset (criar automaticamente)
├── src/
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── ...
├── models/                         # Modelos salvos
├── results/                        # Resultados e gráficos
└── notebooks/                      # Jupyter notebooks
```

---

## 🎮 Exemplo de Sessão Completa

```batch
REM 1. Setup inicial (primeira vez)
download_dataset_windows.bat
REM Aguardar download e organização (~5-10 min)

REM 2. Iniciar ambiente
start_windows.bat
REM Container deve iniciar em ~30 segundos

REM 3. Verificar GPU
docker compose exec tuberculosis-detection-gpu nvidia-smi
REM Deve mostrar sua GPU NVIDIA

REM 4. Teste rápido (1 época)
docker compose exec tuberculosis-detection-gpu python src/train.py ^
    --model resnet50 ^
    --epochs 1 ^
    --batch-size 16

REM 5. Se funcionou, treinar completo
train_all_windows.bat
REM Aguardar conclusão (~2-3h com RTX 3080)

REM 6. Avaliar resultados
docker compose exec tuberculosis-detection-gpu python src/evaluate.py

REM 7. Ver resultados
explorer .\results

REM 8. Parar ambiente
docker compose down
```

---

## 🔗 Links Úteis

- **NVIDIA Drivers**: https://www.nvidia.com/drivers
- **Docker Desktop**: https://www.docker.com/products/docker-desktop
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **Dataset**: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets
- **PyTorch CUDA**: https://pytorch.org/get-started/locally/

---

## ✅ Checklist de Instalação

- [ ] NVIDIA GPU Drivers instalados (`nvidia-smi` funciona)
- [ ] Docker Desktop instalado e rodando
- [ ] WSL 2 habilitado e configurado
- [ ] NVIDIA Container Toolkit instalado
- [ ] Teste de GPU funciona (`docker run --gpus all ...`)
- [ ] Scripts `.bat` na pasta do projeto
- [ ] Dataset baixado e organizado
- [ ] Container iniciado com sucesso
- [ ] GPU detectada dentro do container

---

## 🎯 Próximos Passos

1. ✅ Verificar pré-requisitos
2. ✅ Executar `download_dataset_windows.bat`
3. ✅ Executar `start_windows.bat`
4. ✅ Acessar http://localhost:8888
5. ✅ Testar com 1 época
6. ✅ Treinar modelos completos
7. ✅ Avaliar resultados

**Tudo pronto para começar!** 🚀
