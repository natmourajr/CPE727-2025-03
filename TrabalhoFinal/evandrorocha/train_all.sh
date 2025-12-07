#!/bin/bash

# Script para treinar modelos

echo "🎯 Iniciando treinamento de modelos..."

# Lista de modelos para treinar
MODELS=("resnet50" "densenet121" "efficientnet_b0")

for MODEL in "${MODELS[@]}"
do
    echo ""
    echo "📈 Treinando $MODEL..."
    docker-compose exec tuberculosis-detection python src/train.py --model $MODEL
done

echo ""
echo "✅ Treinamento concluído!"
echo "📊 Resultados salvos em ./models/ e ./results/"
