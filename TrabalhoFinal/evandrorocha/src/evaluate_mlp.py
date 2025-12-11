"""
Script para Avaliar MLP Baseline no Conjunto de Teste
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import json
import sys
import os
from pathlib import Path

sys.path.append('.')
from src.train_mlp_baseline import extract_features_from_dataset

# Importar MLP
try:
    from models.mlp import SimpleMLP
except ImportError:
    sys.path.append('models')
    from mlp import SimpleMLP


def evaluate_mlp(model_path, data_dir='./data/shenzhen', save_path='models/mlp_baseline_test_metrics.json'):
    """
    Avalia MLP no conjunto de teste
    
    Args:
        model_path: Caminho para o modelo treinado
        data_dir: Diretório do dataset
        save_path: Onde salvar as métricas
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Carregar modelo
    print(f"\n🔄 Carregando modelo de: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SimpleMLP(
        input_size=81, 
        hidden_size=128, 
        num_classes=2,
        dropout_rate=0.3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modelo carregado (época {checkpoint.get('epoch', 'N/A')})")
    
    # Extrair features do teste
    print("\n🔄 Extraindo features do conjunto de teste...")
    test_features, test_labels = extract_features_from_dataset(data_dir, 'test')
    
    print(f"✅ Features extraídas: {test_features.shape}")
    
    # Carregar scaler do treinamento (CRÍTICO: Não fit no teste!)
    print("\n🔄 Carregando scaler do treinamento...")
    model_dir = Path(model_path).parent
    scaler_path = model_dir / 'scaler.pkl'
    
    if not scaler_path.exists():
        print(f"⚠️  AVISO: Scaler não encontrado em {scaler_path}")
        print("   Usando novo scaler (pode causar data leakage!)")
        scaler = StandardScaler()
        test_features = scaler.fit_transform(test_features)
    else:
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        test_features = scaler.transform(test_features)  # ← Apenas transform!
        print(f"✅ Scaler carregado de: {scaler_path}")
    
    # Predições
    print("\n🔄 Fazendo predições...")
    with torch.no_grad():
        features_tensor = torch.FloatTensor(test_features).to(device)
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    preds = preds.cpu().numpy()
    probs = probs[:, 1].cpu().numpy()
    
    # Calcular métricas
    print("\n📊 Calculando métricas...")
    cm = confusion_matrix(test_labels, preds)
    
    metrics = {
        'loss': 0.0,  # Não calculamos loss aqui
        'accuracy': float(accuracy_score(test_labels, preds)),
        'precision': float(precision_score(test_labels, preds, zero_division=0)),
        'recall': float(recall_score(test_labels, preds, zero_division=0)),
        'f1_score': float(f1_score(test_labels, preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(test_labels, probs)),
        'confusion_matrix': cm.tolist()
    }
    
    # Calcular sensibilidade e especificidade
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Salvar
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Imprimir resultados
    print("\n" + "="*70)
    print("RESULTADOS DO MLP BASELINE")
    print("="*70)
    print(f"\n📊 Métricas Gerais:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"  AUC-ROC:   {metrics['auc_roc']*100:.2f}%")
    
    print(f"\n🎯 Métricas Clínicas:")
    print(f"  Sensibilidade: {sensitivity*100:.2f}%")
    print(f"  Especificidade: {specificity*100:.2f}%")
    
    print(f"\n📋 Matriz de Confusão:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    
    print(f"\n✅ Métricas salvas em: {save_path}")
    print("="*70)
    
    return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Avaliar MLP Baseline')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Caminho para o modelo treinado (.pth)')
    parser.add_argument('--data-dir', type=str, default='./data/shenzhen',
                        help='Diretório do dataset')
    parser.add_argument('--save-path', type=str, default='models/mlp_baseline_test_metrics.json',
                        help='Onde salvar as métricas')
    
    args = parser.parse_args()
    
    # Verificar se arquivo existe
    if not Path(args.model_path).exists():
        print(f"❌ Erro: Modelo não encontrado em {args.model_path}")
        print("\nProcurando modelos MLP...")
        mlp_dirs = list(Path('results').glob('mlp_baseline_*'))
        if mlp_dirs:
            print("\nModelos encontrados:")
            for d in mlp_dirs:
                model_file = d / 'best_model.pth'
                if model_file.exists():
                    print(f"  ✓ {model_file}")
        return
    
    evaluate_mlp(args.model_path, args.data_dir, args.save_path)


if __name__ == '__main__':
    main()
