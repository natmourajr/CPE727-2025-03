# 🔄 Guia: Download Resumível do Dataset

## ✨ Funcionalidades Implementadas

O script `src/download_data.py` agora possui **suporte a download resumível (resume download)**:

### ✅ O que foi adicionado:

1. **Download Resumível** - Retoma downloads interrompidos automaticamente
2. **Arquivo Parcial** - Salva progresso em `.part` durante o download
3. **Limpeza de Downloads** - Opção para limpar e recomeçar
4. **Detecção Inteligente** - Verifica se arquivo já existe antes de baixar
5. **Tratamento de Erros** - Melhor handling de falhas de conexão

---

## 🎯 Cenários de Uso

### 1. Download Normal (Primeira Vez)

```bash
# Mac M1
docker compose exec tuberculosis-detection-m1 python src/download_data.py

# Windows GPU
docker compose exec tuberculosis-detection-gpu python src/download_data.py
```

**O que acontece:**
- ✅ Baixa `shenzhen_dataset.zip`
- ✅ Salva progresso em `shenzhen_dataset.zip.part` durante download
- ✅ Renomeia para `.zip` quando completo
- ✅ Extrai e organiza automaticamente

---

### 2. Download Interrompido (Conexão Caiu)

```bash
# Execute novamente o mesmo comando
docker compose exec tuberculosis-detection-m1 python src/download_data.py
```

**O que acontece:**
- 🔍 Detecta arquivo `.part` existente
- 📦 Mostra tamanho já baixado (ex: "145.2 MB")
- 🔄 **Retoma o download de onde parou**
- ✅ Completa apenas o que falta

**Exemplo de saída:**
```
📦 Download parcial encontrado: 145.2 MB
🔄 Retomando download...
shenzhen_dataset.zip: 65%|████████████     | 287MB/440MB [02:15<01:08, 2.24MB/s]
```

---

### 3. Download Pausado Manualmente (Ctrl+C)

```bash
# Durante o download, pressione Ctrl+C
^C
⚠️  Download interrompido pelo usuário
💾 Download parcial salvo em: data/shenzhen_dataset.zip.part
🔄 Execute novamente para retomar o download

# Quando quiser retomar:
docker compose exec tuberculosis-detection-m1 python src/download_data.py
```

---

### 4. Limpar e Recomeçar do Zero

```bash
# Limpar apenas arquivo parcial (.part)
docker compose exec tuberculosis-detection-m1 python src/download_data.py --clean

# Limpar tudo e forçar re-download
docker compose exec tuberculosis-detection-m1 python src/download_data.py --clean --force

# Depois baixar novamente
docker compose exec tuberculosis-detection-m1 python src/download_data.py
```

---

### 5. Arquivo Já Existe (Pular Download)

```bash
docker compose exec tuberculosis-detection-m1 python src/download_data.py
```

**Saída:**
```
✅ Arquivo já existe: data/shenzhen_dataset.zip
📦 Pulando download e indo direto para extração...
```

Se quiser forçar re-download:
```bash
docker compose exec tuberculosis-detection-m1 python src/download_data.py --clean --force
docker compose exec tuberculosis-detection-m1 python src/download_data.py
```

---

## 🔧 Opções da Linha de Comando

```bash
# Ajuda
python src/download_data.py --help

# Especificar diretório de saída
python src/download_data.py --output-dir /caminho/personalizado

# Apenas verificar dataset (não baixa)
python src/download_data.py --verify-only

# Organizar dataset baixado manualmente
python src/download_data.py --organize-only --source /caminho/extraido

# Limpar downloads parciais
python src/download_data.py --clean

# Forçar re-download completo
python src/download_data.py --clean --force
python src/download_data.py
```

---

## 📊 Exemplos Práticos

### Exemplo 1: Download com Falha de Conexão

```bash
# Tentativa 1 (falhou em 30%)
$ docker compose exec tuberculosis-detection-m1 python src/download_data.py
📥 Tentando baixar dataset automaticamente...
shenzhen_dataset.zip: 30%|███         | 132MB/440MB [01:30<03:30, 1.5MB/s]
❌ Erro no download: Connection reset by peer
💾 Download parcial salvo em: data/shenzhen_dataset.zip.part
🔄 Execute novamente para retomar o download

# Tentativa 2 (retoma de 30%)
$ docker compose exec tuberculosis-detection-m1 python src/download_data.py
📦 Download parcial encontrado: 132.0 MB
🔄 Retomando download...
shenzhen_dataset.zip: 100%|████████████| 440MB/440MB [04:30<00:00, 1.6MB/s]
✅ Download concluído com sucesso!
```

---

### Exemplo 2: Múltiplas Interrupções

```bash
# 1ª tentativa (20%)
$ docker compose exec tuberculosis-detection-m1 python src/download_data.py
# Ctrl+C
⚠️  Download interrompido pelo usuário
💾 Download parcial: 88 MB

# 2ª tentativa (40%)
$ docker compose exec tuberculosis-detection-m1 python src/download_data.py
📦 Download parcial encontrado: 88.0 MB
🔄 Retomando...
# Ctrl+C novamente
💾 Download parcial: 176 MB

# 3ª tentativa (completa)
$ docker compose exec tuberculosis-detection-m1 python src/download_data.py
📦 Download parcial encontrado: 176.0 MB
🔄 Retomando...
✅ Download concluído com sucesso!
```

---

### Exemplo 3: Servidor não Suporta Resumo

```bash
$ docker compose exec tuberculosis-detection-m1 python src/download_data.py
📦 Download parcial encontrado: 200.0 MB
🔄 Retomando download...
⚠️  Servidor não suporta resumo, baixando do início...
shenzhen_dataset.zip: 100%|████████████| 440MB/440MB [05:00<00:00, 1.5MB/s]
```

---

## 🛡️ Tratamento de Erros

### Erro de Conexão
```python
❌ Erro no download: Connection reset by peer
💾 Download parcial salvo em: data/shenzhen_dataset.zip.part
🔄 Execute novamente para retomar o download
```
**Solução:** Execute novamente, o download retomará automaticamente.

---

### Timeout
```python
❌ Erro no download: Read timed out
💾 Download parcial salvo: 250 MB
🔄 Execute novamente para retomar
```
**Solução:** Execute novamente com conexão estável.

---

### Interrupção Manual
```python
⚠️  Download interrompido pelo usuário
💾 Download parcial salvo em: data/shenzhen_dataset.zip.part
🔄 Execute novamente para retomar o download
```
**Solução:** Execute novamente quando estiver pronto.

---

### Espaço em Disco Insuficiente
```python
❌ Erro inesperado: [Errno 28] No space left on device
```
**Solução:** Libere espaço em disco e execute novamente (retomará do ponto atual).

---

## 📁 Estrutura de Arquivos

Durante o download:
```
data/
├── shenzhen_dataset.zip.part    # Download em progresso
└── shenzhen/                    # Não existe ainda
```

Após download completo:
```
data/
├── shenzhen_dataset.zip         # Arquivo completo
└── shenzhen/                    # Extraído e organizado
    ├── normal/
    └── tuberculosis/
```

Após organização:
```
data/
└── shenzhen/                    # ZIP e temporários são limpos
    ├── normal/          # 326 imagens
    └── tuberculosis/    # 240 imagens
```

---

## 💡 Dicas

### 1. Monitorar Progresso
```bash
# Em outro terminal
docker compose exec tuberculosis-detection-m1 ls -lh data/*.part

# Ver tamanho do arquivo parcial
watch -n 1 'ls -lh data/*.zip*'
```

### 2. Conexão Instável
Se sua conexão cai frequentemente:
```bash
# Execute em loop até completar
while ! docker compose exec tuberculosis-detection-m1 python src/download_data.py; do
    echo "Retentando em 5 segundos..."
    sleep 5
done
```

### 3. Download Manual Alternativo
Se o resumo não funcionar:
1. Baixe manualmente do site
2. Coloque em `data/shenzhen_dataset.zip`
3. Execute: `python src/download_data.py` (pulará download e irá direto para extração)

---

## 🔍 Verificar Estado Atual

```bash
# Verificar se há download parcial
docker compose exec tuberculosis-detection-m1 ls -lh data/*.part 2>/dev/null

# Verificar se download está completo
docker compose exec tuberculosis-detection-m1 ls -lh data/shenzhen_dataset.zip

# Verificar dataset organizado
docker compose exec tuberculosis-detection-m1 python src/download_data.py --verify-only
```

---

## ⚙️ Como Funciona (Técnico)

### 1. HTTP Range Requests
O script usa **HTTP Range headers** para retomar downloads:
```python
headers = {'Range': f'bytes={downloaded_size}-'}
response = requests.get(url, headers=headers)
```

### 2. Status Code 206 (Partial Content)
Servidor responde com `206 Partial Content` quando suporta resumo:
```python
if response.status_code == 206:
    # Servidor suporta resumo
    mode = 'ab'  # Append mode
else:
    # Servidor não suporta, recomeça
    mode = 'wb'  # Write mode
```

### 3. Arquivo Temporário `.part`
Durante download, salva em `.part`:
```python
temp_file = destination.with_suffix(destination.suffix + '.part')
# Ao completar, renomeia:
temp_file.rename(destination)
```

---

## ✅ Checklist de Recuperação

- [ ] Download interrompido? → Execute novamente
- [ ] Conexão instável? → Use loop de retry
- [ ] Servidor não suporta resumo? → Download manual
- [ ] Arquivo corrompido? → `--clean --force` e baixe novamente
- [ ] Espaço em disco? → Libere espaço, retome automaticamente

---

**Resumo:** Agora você pode **pausar e retomar** downloads sem perder progresso! 🎉
