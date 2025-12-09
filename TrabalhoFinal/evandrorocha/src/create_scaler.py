"""
Script para criar scaler do treinamento e re-avaliar MLP
"""
import sys
sys.path.append('.')

from src.train_mlp_baseline import extract_features_from_dataset
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# Extrair features do treino
print("Extraindo features do treino...")
train_features, train_labels = extract_features_from_dataset('./data/shenzhen', 'train')

# Criar e treinar scaler
print("Criando scaler...")
scaler = StandardScaler()
scaler.fit(train_features)

# Salvar
model_dir = Path('results/mlp_baseline_20251208_213613')
scaler_path = model_dir / 'scaler.pkl'

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Scaler salvo em: {scaler_path}")
print("\nAgora execute:")
print("docker compose exec tuberculosis-detection-gpu python src/evaluate_mlp.py \\")
print("  --model-path results/mlp_baseline_20251208_213613/best_model.pth")
