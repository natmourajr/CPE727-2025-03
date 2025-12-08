"""
Script de Avaliação para MLP - Detecção de Tuberculose
Avalia modelo treinado e gera relatório completo com métricas
"""

import os
import sys
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP_TB_Complete, MLP_TB_Detector, FeatureExtractorCNN


def plot_confusion_matrix(cm, classes, save_path):
    """Plota matriz de confusão"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")


def plot_roc_curve(fpr, tpr, auc, save_path):
    """Plota curva ROC"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    plt.title('Curva ROC - Detecção de Tuberculose')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curva ROC salva em: {save_path}")


def evaluate_model(model, dataloader, device, class_names):
    """
    Avalia o modelo e retorna todas as métricas
    
    Returns:
        dict com todas as métricas
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Avaliando modelo...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob da classe TB
    
    # Converte para numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calcula métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Sensitivity e Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names,
                                   output_dict=True)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        },
        'classification_report': report
    }
    
    return results, (fpr, tpr, auc), cm


def print_results(results, class_names):
    """Imprime resultados formatados"""
    print("\n" + "="*60)
    print("RESULTADOS DA AVALIAÇÃO")
    print("="*60)
    
    print("\n📊 Métricas Gerais:")
    print(f"  Acurácia:        {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precisão:        {results['precision']:.4f}")
    print(f"  Recall:          {results['recall']:.4f}")
    print(f"  F1-Score:        {results['f1_score']:.4f}")
    print(f"  AUC-ROC:         {results['auc_roc']:.4f}")
    
    print("\n🎯 Métricas Clínicas:")
    print(f"  Sensibilidade:   {results['sensitivity']:.4f} ({results['sensitivity']*100:.2f}%)")
    print(f"  Especificidade:  {results['specificity']:.4f} ({results['specificity']*100:.2f}%)")
    
    print("\n📋 Matriz de Confusão:")
    cm = np.array(results['confusion_matrix'])
    print(f"                 Predito {class_names[0]}  Predito {class_names[1]}")
    print(f"  Real {class_names[0]}:        {cm[0,0]:4d}          {cm[0,1]:4d}")
    print(f"  Real {class_names[1]}:        {cm[1,0]:4d}          {cm[1,1]:4d}")
    
    print("\n✅ Verdadeiros Positivos (TP):  ", results['true_positives'])
    print("✅ Verdadeiros Negativos (TN):  ", results['true_negatives'])
    print("❌ Falsos Positivos (FP):       ", results['false_positives'])
    print("❌ Falsos Negativos (FN):       ", results['false_negatives'])
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Avaliar MLP para Detecção de TB')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Caminho para o checkpoint do modelo')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Diretório com dados de teste')
    parser.add_argument('--mode', type=str, default='two_stage',
                       choices=['two_stage', 'end_to_end'],
                       help='Modo do modelo')
    parser.add_argument('--features-path', type=str, default=None,
                       help='Caminho para features pré-extraídas (two_stage)')
    parser.add_argument('--save-dir', type=str, default='../results/evaluation',
                       help='Diretório para salvar resultados')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamanho do batch')
    
    args = parser.parse_args()
    
    # Cria diretório de resultados
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Carrega modelo
    print(f"\nCarregando modelo de: {args.checkpoint}")
    
    if args.mode == 'two_stage':
        # Carrega features se fornecidas
        if args.features_path:
            print(f"Carregando features de: {args.features_path}")
            test_features = np.load(os.path.join(args.features_path, 'test_features.npy'))
            test_labels = np.load(os.path.join(args.features_path, 'test_labels.npy'))
            
            test_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(test_features),
                    torch.LongTensor(test_labels)
                ),
                batch_size=args.batch_size,
                shuffle=False
            )
            
            model = MLP_TB_Detector()
        else:
            # Precisa extrair features primeiro
            print("Extraindo features do conjunto de teste...")
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            test_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
            extract_loader = DataLoader(
                test_dataset, batch_size=args.batch_size * 2,
                shuffle=False, num_workers=4
            )
            
            # Extrator
            feature_extractor = FeatureExtractorCNN(pretrained=True, freeze_layers=True)
            feature_extractor = feature_extractor.to(device)
            feature_extractor.eval()
            
            all_features = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(extract_loader, desc="Extracting features"):
                    images = images.to(device)
                    features = feature_extractor(images)
                    all_features.append(features.cpu().numpy())
                    all_labels.append(labels.numpy())
            
            test_features = np.vstack(all_features)
            test_labels = np.concatenate(all_labels)
            
            # Salva features
            np.save(os.path.join(args.save_dir, 'test_features.npy'), test_features)
            np.save(os.path.join(args.save_dir, 'test_labels.npy'), test_labels)
            
            test_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(test_features),
                    torch.LongTensor(test_labels)
                ),
                batch_size=args.batch_size,
                shuffle=False
            )
            
            model = MLP_TB_Detector()
    
    else:  # end_to_end
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4
        )
        
        model = MLP_TB_Complete()
    
    # Carrega pesos
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo carregado da época {checkpoint.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Classes
    if args.mode == 'two_stage' and args.features_path:
        class_names = ['Normal', 'TB']
    else:
        class_names = test_dataset.classes
    
    print(f"Classes: {class_names}")
    print(f"Total de amostras de teste: {len(test_loader.dataset)}")
    
    # Avalia
    results, roc_data, cm = evaluate_model(model, test_loader, device, class_names)
    
    # Imprime resultados
    print_results(results, class_names)
    
    # Salva resultados
    results_path = os.path.join(args.save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResultados salvos em: {results_path}")
    
    # Plota matriz de confusão
    cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Plota curva ROC
    fpr, tpr, auc = roc_data
    roc_path = os.path.join(args.save_dir, 'roc_curve.png')
    plot_roc_curve(fpr, tpr, auc, roc_path)
    
    print("\n✅ Avaliação concluída com sucesso!")


if __name__ == '__main__':
    main()
