#!/bin/bash

echo "=========================================="
echo "DETECÇÃO AUTOMÁTICA DO SISTEMA"
echo "=========================================="

# Detectar sistema operacional e arquitetura
OS=$(uname -s)
ARCH=$(uname -m)

echo "Sistema: $OS"
echo "Arquitetura: $ARCH"
echo ""

# Determinar qual profile usar
PROFILE=""
CONTAINER_NAME=""

if [[ "$OS" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
    echo "✅ Detectado: Mac Apple Silicon (M1/M2/M3)"
    PROFILE="m1"
    CONTAINER_NAME="tuberculosis-detection-m1"
    echo "🚀 Usando profile: $PROFILE (ARM64, CPU/MPS)"
    
elif [[ "$OS" == "Darwin" ]] && [[ "$ARCH" == "x86_64" ]]; then
    echo "✅ Detectado: Mac Intel"
    PROFILE="cpu"
    CONTAINER_NAME="tuberculosis-detection-cpu"
    echo "🚀 Usando profile: $PROFILE (x86_64, CPU)"
    
elif [[ "$OS" == "Linux" ]]; then
    echo "✅ Detectado: Linux"
    
    # Verificar se tem GPU NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ $GPU_COUNT -gt 0 ]; then
            echo "✅ GPU NVIDIA detectada:"
            nvidia-smi --query-gpu=name --format=csv,noheader
            PROFILE="gpu"
            CONTAINER_NAME="tuberculosis-detection-gpu"
            echo "🚀 Usando profile: $PROFILE (NVIDIA CUDA)"
        else
            echo "⚠️  nvidia-smi encontrado mas nenhuma GPU detectada"
            PROFILE="cpu"
            CONTAINER_NAME="tuberculosis-detection-cpu"
            echo "🚀 Usando profile: $PROFILE (CPU apenas)"
        fi
    else
        echo "⚠️  GPU NVIDIA não detectada"
        PROFILE="cpu"
        CONTAINER_NAME="tuberculosis-detection-cpu"
        echo "🚀 Usando profile: $PROFILE (CPU apenas)"
    fi
else
    echo "❌ Sistema não reconhecido: $OS $ARCH"
    echo ""
    echo "Execute manualmente com um dos profiles:"
    echo "  Mac Apple Silicon:   COMPOSE_PROFILES=m1 docker compose up"
    echo "  Intel/AMD com GPU:   COMPOSE_PROFILES=gpu docker compose up"
    echo "  Intel/AMD sem GPU:   COMPOSE_PROFILES=cpu docker compose up"
    exit 1
fi

echo ""
echo "─────────────────────────────────────────"
echo ""

# Parar containers existentes
#echo "🛑 Parando containers existentes..."
#docker compose down 2>/dev/null

echo ""
echo "🔨 Construindo imagem Docker..."
COMPOSE_PROFILES=$PROFILE docker compose build --no-cache

if [ $? -ne 0 ]; then
    echo "❌ Erro ao construir imagem!"
    exit 1
fi

echo ""
echo "🚀 Iniciando container com profile: $PROFILE"
COMPOSE_PROFILES=$PROFILE docker compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Erro ao iniciar container!"
    exit 1
fi

# Aguardar container iniciar
echo ""
echo "⏳ Aguardando container iniciar..."
sleep 3

echo ""
echo "✅ Container iniciado com sucesso!"
echo ""
echo "─────────────────────────────────────────"
echo "📍 JUPYTER LAB: http://localhost:8888"
echo "─────────────────────────────────────────"
echo ""
echo "Comandos úteis:"
echo "  Ver logs:          docker compose logs -f"
echo "  Parar:            docker compose down"
echo "  Entrar:           docker compose exec $CONTAINER_NAME bash"
echo "  Reiniciar:        docker compose restart"
echo ""

# Exibir informações específicas do profile
case $PROFILE in
    "m1")
        echo "⚠️  NOTA: Mac Apple Silicon usa aceleração MPS"
        echo "   (mais lento que GPU CUDA, mas mais rápido que CPU puro)"
        echo "   Recomendado: --batch-size 8"
        ;;
    "gpu")
        echo "🎮 GPU NVIDIA disponível para treinamento acelerado"
        echo "   Recomendado: --batch-size 16 ou 32"
        ;;
    "cpu")
        echo "⚠️  NOTA: Treinamento em CPU (mais lento)"
        echo "   Recomendado: --batch-size 8"
        ;;
esac

echo ""