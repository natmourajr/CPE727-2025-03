# 🚀 Início Rápido - Windows + GPU NVIDIA

## ⚡ 3 Comandos para Começar

```batch
download_dataset_windows.bat    # 1. Download dataset
start_windows.bat               # 2. Iniciar ambiente  
train_all_windows.bat          # 3. Treinar modelos
```

## 📋 Pré-requisitos (5 minutos)

### 1. NVIDIA Drivers
```powershell
nvidia-smi  # Deve mostrar sua GPU
```
Se não funcionar: https://www.nvidia.com/drivers

### 2. Docker Desktop
Download: https://www.docker.com/products/docker-desktop
- ✅ Habilitar WSL 2

### 3. NVIDIA Container Toolkit
```bash
# No WSL (Ubuntu):
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 4. Testar GPU
```powershell
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```
✅ Deve mostrar informações da GPU

---

## 🎯 Uso Básico

### Iniciar Ambiente
```batch
start_windows.bat
```
Acesse: **http://localhost:8888**

### Treinar Modelo
```powershell
docker compose exec tuberculosis-detection-gpu python src/train.py ^
    --model resnet50 ^
    --epochs 50 ^
    --batch-size 32
```

### Verificar GPU
```powershell
docker compose exec tuberculosis-detection-gpu nvidia-smi
```

### Parar
```batch
docker compose down
```

---

## 📊 Performance

| GPU | Batch Size | Tempo/Época |
|-----|------------|-------------|
| RTX 3060 | 16 | ~4-5 min |
| RTX 3080 | 32 | ~2-3 min |
| RTX 4090 | 64 | ~1 min |

**vs Mac M1**: 5-10x mais rápido! 🚀

---

## 🐛 Problemas Comuns

### GPU não detectada
```powershell
# Verificar
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Reinstalar toolkit
sudo apt-get install --reinstall nvidia-docker2
sudo systemctl restart docker
```

### Out of Memory
```powershell
# Reduzir batch size
docker compose exec tuberculosis-detection-gpu python src/train.py --batch-size 8
```

### Porta 8888 ocupada
```powershell
netstat -ano | findstr :8888
taskkill /PID <PID> /F
```

---

## 📚 Documentação Completa

Ver: **WINDOWS_GUIDE.md**

---

## ✅ Checklist

- [ ] `nvidia-smi` funciona
- [ ] Docker Desktop instalado
- [ ] WSL 2 habilitado
- [ ] NVIDIA Container Toolkit instalado
- [ ] Teste GPU funciona
- [ ] Scripts `.bat` prontos

**Tudo OK? Execute `start_windows.bat`!** 🎮
