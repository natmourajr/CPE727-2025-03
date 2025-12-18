# 🐳 Como Rodar o Projeto com Docker

Este guia contém as instruções passo a passo para executar o ambiente de desenvolvimento, treinamento e avaliação usando Docker.

## ✅ Pré-requisitos

1.  **Docker Desktop** instalado e rodando.
2.  **(Opcional) Drivers NVIDIA** atualizados (para uso de GPU).
    *   *Nota: O projeto detecta automaticamente se você tem GPU ou CPU.*

---

## 🚀 1. Iniciar o Ambiente

Abra o terminal na pasta `TrabalhoFinal/evandrorocha` e execute:

### 🪟 Windows (Powershell/CMD)
```powershell
.\start_windows.bat
```

### 🐧 Linux / 🍎 Mac
```bash
chmod +x start.sh
./start.sh
```

> **O que isso faz?**
> *   Constrói a imagem Docker (se necessário).
> *   Inicia o container `tuberculosis-detection`.
> *   Sobe o servidor Jupyter Lab.

---

## 📥 2. Baixar o Dataset

Se é a primeira vez rodando, você precisa baixar as imagens de Raio-X.

```bash
# Windows
docker-compose run --rm tuberculosis-detection python src/download_data.py

# Linux/Mac
./download_dataset.sh
```

---

## 🧠 3. Treinar os Modelos

Para treinar todos os modelos (ResNet, DenseNet, EfficientNet, SimpleCNN):

```bash
# Windows
.\train_all_windows.bat

# Linux/Mac
./train_all.sh
```

Para treinar **apenas um modelo específico** (ex: ResNet50):

```bash
docker-compose exec tuberculosis-detection python src/train.py --model resnet50
```

---

## 📊 4. Avaliar Resultados

Para gerar as métricas, matrizes de confusão e gráficos comparativos:

```bash
docker-compose exec tuberculosis-detection python src/evaluate.py
```

Os resultados serão salvos na pasta `results/`.

---

## 📓 5. Acessar Notebooks (Jupyter)

O Jupyter Lab fica disponível automaticamente após o início do ambiente.

*   **URL:** [http://localhost:8888](http://localhost:8888)
*   **Token:** (Geralmente não é necessário, ou verifique no terminal se solicitado)

---

## 🛠️ Comandos Úteis

| Ação | Comando |
| :--- | :--- |
| **Parar tudo** | `docker-compose down` |
| **Ver logs** | `docker-compose logs -f` |
| **Entrar no terminal do container** | `docker-compose exec tuberculosis-detection bash` |
| **Reconstruir imagem** | `docker-compose build --no-cache` |

---

## ❓ Problemas Comuns

**Erro: "GPU not found"**
*   Verifique se o Docker Desktop está configurado para usar o backend WSL2 (no Windows).
*   Se não tiver GPU, o script usará CPU automaticamente (será mais lento).

**Erro de Permissão (Linux)**
*   Use `sudo` antes dos comandos docker se seu usuário não estiver no grupo docker.
