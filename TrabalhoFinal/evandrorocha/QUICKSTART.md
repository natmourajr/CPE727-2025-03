# Guia Rápido de Início

# Guia Rápido de Início

## 🚀 Setup Rápido (5 minutos)

### 1. Baixar Dataset

**Opção A - Automático (Recomendado):**
```bash
# Um único comando faz tudo!
./download_dataset.sh
```

**Opção B - Manual:**
```bash
# 1. Baixe de: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets
# 2. Procure "Shenzhen Hospital X-ray Set" e baixe ChinaSet_AllFiles.zip
# 3. Execute:
docker-compose build
docker-compose run --rm tuberculosis-detection python src/download_data.py
```

**Verificar:**
```bash
docker-compose run --rm tuberculosis-detection python src/download_data.py --verify-only
```

✅ **Você deve ver:** 326 imagens normais + 240 com TB = 566 total

### 2. Iniciar Container Docker
```bash
# Opção 1: Usando script
./start.sh

# Opção 2: Manual
docker-compose up -d
```

### 3. Acessar Jupyter Lab
Abra o navegador em: http://localhost:8888

### 4. Explorar Dados
Abra o notebook: `notebooks/01_data_exploration.ipynb`

### 5. Treinar Modelo
```bash
# No terminal ou dentro do container
python src/train.py
```

## 📋 Checklist

- [ ] Docker instalado
- [ ] Dataset baixado e organizado
- [ ] Container iniciado
- [ ] Jupyter Lab acessível
- [ ] GPU detectada (opcional, mas recomendado)

## ⚡ Comandos Essenciais

```bash
# Ver logs
docker-compose logs -f

# Parar container
docker-compose down

# Entrar no container
docker-compose exec tuberculosis-detection bash

# Treinar modelo específico
docker-compose exec tuberculosis-detection python src/train.py

# Avaliar modelos
docker-compose exec tuberculosis-detection python src/evaluate.py
```

## 🐛 Problemas Comuns

### Dataset não encontrado
```bash
# Verifique a estrutura
ls -la data/shenzhen/normal
ls -la data/shenzhen/tuberculosis
```

### Sem GPU
Edite `docker-compose.yml` e remova a seção `deploy`.

### Erro de memória
Reduza `BATCH_SIZE` em `src/train.py`.

## 📊 Próximos Passos

1. ✅ Explorar dados (notebook)
2. ✅ Treinar primeiro modelo
3. ✅ Avaliar resultados
4. ✅ Comparar diferentes arquiteturas
5. ✅ Ajustar hiperparâmetros
6. ✅ Gerar relatório final
