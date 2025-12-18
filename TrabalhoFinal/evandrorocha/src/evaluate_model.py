"""
Script de Avaliação Completa do Modelo ResNet50
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from torch.utils.data import DataLoader
import sys
sys.path.append('src')

from dataset import get_dataloaders
from models import create_model


def load_model(model_path, device='cuda'):
    """Carrega modelo treinado"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_model(
        checkpoint.get('model_name', 'resnet50'),
        pretrained=False,
        num_classes=2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, test_loader, device='cuda'):
    """Avalia modelo no conjunto de teste"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'TB'],
                yticklabels=['Normal', 'TB'])
    plt.title('Matriz de Confusão - ResNet50', fontsize=14, fontweight='bold')
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Predito', fontsize=12)
    
    # Adicionar percentuais
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Matriz de confusão salva em: {save_path}')
    
    return cm


def plot_roc_curve(y_true, y_probs, save_path=None):
    """Plota curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title('Curva ROC - ResNet50', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Curva ROC salva em: {save_path}')
    
    return fpr, tpr, roc_auc


def plot_precision_recall_curve(y_true, y_probs, save_path=None):
    """Plota curva Precision-Recall"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall (Sensibilidade)', fontsize=12)
    plt.ylabel('Precision (Precisão)', fontsize=12)
    plt.title('Curva Precision-Recall - ResNet50', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Curva Precision-Recall salva em: {save_path}')


def print_detailed_report(y_true, y_pred, y_probs):
    """Imprime relatório detalhado"""
    print("="*70)
    print("RELATÓRIO DE AVALIAÇÃO - RESNET50")
    print("="*70)
    
    # Classification report
    print("\n📊 Relatório de Classificação:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Tuberculosis'],
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📋 Matriz de Confusão:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    
    # Métricas clínicas
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n🎯 Métricas Clínicas:")
    print(f"  Sensibilidade (Recall): {sensitivity*100:.2f}%")
    print(f"  Especificidade:         {specificity*100:.2f}%")
    
    # ROC-AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_true, y_probs)
    print(f"  AUC-ROC:                {roc_auc*100:.2f}%")
    
    print("\n💡 Interpretação:")
    if sensitivity >= 0.9:
        print(f"  ✅ Sensibilidade EXCELENTE ({sensitivity*100:.1f}%)")
    elif sensitivity >= 0.8:
        print(f"  ✓ Sensibilidade BOA ({sensitivity*100:.1f}%)")
    else:
        print(f"  ⚠️ Sensibilidade BAIXA ({sensitivity*100:.1f}%)")
    
    if specificity >= 0.9:
        print(f"  ✅ Especificidade EXCELENTE ({specificity*100:.1f}%)")
    elif specificity >= 0.8:
        print(f"  ✓ Especificidade BOA ({specificity*100:.1f}%)")
    else:
        print(f"  ⚠️ Especificidade BAIXA ({specificity*100:.1f}%)")
    
    print("="*70)


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Avaliar modelo ResNet50')
    parser.add_argument('--model-path', type=str, default='./models/resnet50_best.pth',
                        help='Caminho para o modelo')
    parser.add_argument('--data-dir', type=str, default='./data/shenzhen',
                        help='Diretório dos dados')
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Diretório para salvar resultados')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda ou cpu)')
    
    args = parser.parse_args()
    
    # Criar diretório de resultados
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    print("🔄 Carregando modelo...")
    model, checkpoint = load_model(args.model_path, args.device)
    
    print(f"✅ Modelo carregado: {checkpoint.get('model_name', 'resnet50')}")
    print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val F1: {checkpoint.get('metrics', {}).get('f1_score', 'N/A')}")
    
    print("\n🔄 Carregando dados de teste...")
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=16,
        num_workers=4
    )
    
    print(f"✅ {len(test_loader.dataset)} imagens de teste carregadas")
    
    print("\n🔄 Avaliando modelo...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, args.device)
    
    # Relatório detalhado
    print_detailed_report(y_true, y_pred, y_probs)
    
    # Gerar visualizações
    print("\n📊 Gerando visualizações...")
    
    cm = plot_confusion_matrix(
        y_true, y_pred, 
        save_path=f'{args.save_dir}/resnet50_confusion_matrix.png'
    )
    
    fpr, tpr, roc_auc = plot_roc_curve(
        y_true, y_probs,
        save_path=f'{args.save_dir}/resnet50_roc_curve.png'
    )
    
    plot_precision_recall_curve(
        y_true, y_probs,
        save_path=f'{args.save_dir}/resnet50_precision_recall.png'
    )
    
    print("\n✅ Avaliação completa!")
    print(f"📁 Resultados salvos em: {args.save_dir}/")


if __name__ == '__main__':
    main()
