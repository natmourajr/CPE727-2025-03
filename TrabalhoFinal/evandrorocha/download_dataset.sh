#!/bin/bash

echo "=========================================="
echo "DOWNLOAD DO DATASET SHENZHEN"
echo "=========================================="

# Detectar sistema e arquitetura
OS=$(uname -s)
ARCH=$(uname -m)

echo "Sistema: $OS $ARCH"
echo ""

# Determinar profile automaticamente
PROFILE=""
CONTAINER_NAME=""

if [[ "$OS" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
    PROFILE="m1"
    CONTAINER_NAME="tuberculosis-detection-m1"
    echo "✅ Detectado: Mac Apple Silicon"
elif [[ "$OS" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
    PROFILE="gpu"
    CONTAINER_NAME="tuberculosis-detection-gpu"
    echo "✅ Detectado: Linux com GPU"
else
    PROFILE="cpu"
    CONTAINER_NAME="tuberculosis-detection-cpu"
    echo "✅ Detectado: Sistema com CPU"
fi

echo "🚀 Usando profile: $PROFILE"
echo ""

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado. Instale o Docker primeiro."
    exit 1
fi

echo "🔨 Construindo container..."
COMPOSE_PROFILES=$PROFILE docker-compose build

if [ $? -ne 0 ]; then
    echo "❌ Erro ao construir container!"
    exit 1
fi

echo ""
echo "📥 Iniciando download do dataset..."
echo "⚠️  Nota: O download automático pode falhar devido a restrições do site NIH."
echo ""
COMPOSE_PROFILES=$PROFILE docker-compose run --rm $CONTAINER_NAME python src/download_data.py

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Download automático falhou!"
    echo ""
    echo "📋 DOWNLOAD MANUAL:"
    echo "1. Acesse: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets"
    echo "2. Baixe 'Shenzhen Hospital X-ray Set' (ChinaSet_AllFiles.zip)"
    echo "3. Coloque em: ./data/shenzhen_dataset.zip"
    echo "4. Execute novamente este script"
    exit 1
fi

echo ""
echo "🔍 Verificando dataset..."
COMPOSE_PROFILES=$PROFILE docker-compose run --rm $CONTAINER_NAME python src/download_data.py --verify-only

echo ""
echo "✅ Processo concluído!"
echo ""
echo "Próximos passos:"
echo "  1. Inicie o ambiente: ./start.sh"
echo "  2. Acesse o Jupyter: http://localhost:8888"
echo "  3. Ou treine modelos: docker-compose exec $CONTAINER_NAME python src/train.py"
