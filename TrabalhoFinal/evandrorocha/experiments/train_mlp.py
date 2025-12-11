"""
Script de Treinamento para MLP - Detecção de Tuberculose
Suporta dois modos:
1. End-to-end: Treina feature extractor + MLP juntos
2. Two-stage: Extrai features primeiro, depois treina MLP
"""

import os
import sys
import argparse
import time
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP_TB_Complete, MLP_TB_Detector, FeatureExtractorCNN


class MetricsTracker:
    """Rastreia métricas durante o treinamento"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_aucs = []
        self.val_sensitivities = []
        self.val_specificities = []
        self.best_val_auc = 0.0
        self.best_epoch = 0
    
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, 
               val_auc, val_sensitivity, val_specificity):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.val_aucs.append(val_auc)
        self.val_sensitivities.append(val_sensitivity)
        self.val_specificities.append(val_specificity)
        
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = epoch
    
    def plot_metrics(self, save_path):
        """Plota e salva gráficos de métricas"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0, 0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.train_accs, label='Train Acc', marker='o')
        axes[0, 1].plot(self.val_accs, label='Val Acc', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC-ROC
        axes[1, 0].plot(self.val_aucs, label='Val AUC-ROC', marker='d', color='green')
        axes[1, 0].axhline(y=self.best_val_auc, color='r', linestyle='--', 
                          label=f'Best AUC: {self.best_val_auc:.4f}')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC-ROC')
        axes[1, 0].set_title('Validation AUC-ROC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Sensitivity & Specificity
        axes[1, 1].plot(self.val_sensitivities, label='Sensitivity', marker='^')
        axes[1, 1].plot(self.val_specificities, label='Specificity', marker='v')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Sensitivity and Specificity')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Métricas salvas em: {save_path}")


def extract_features(model, dataloader, device):
    """
    Extrai features de todas as imagens usando o feature extractor
    
    Args:
        model: FeatureExtractorCNN
        dataloader: DataLoader com imagens
        device: cuda ou cpu
    
    Returns:
        features: numpy array [N, 2048]
        labels: numpy array [N]
    """
    model.eval()
    all_features = []
    all_labels = []
    
    print("Extraindo features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    print(f"Features extraídas: {features.shape}")
    return features, labels


def train_epoch(model, dataloader, criterion, optimizer, device, is_feature_mode=False):
    """Treina por uma época"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Atualiza barra de progresso
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, is_feature_mode=False):
    """Valida o modelo"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidade da classe TB
    
    # Calcula métricas
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    
    # Sensitivity e Specificity
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return epoch_loss, epoch_acc, epoch_auc, sensitivity, specificity


def train_mlp(args):
    """Função principal de treinamento"""
    
    # Configuração do dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cria diretório de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"mlp_{args.mode}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Salva argumentos
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Transformações para as imagens
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Carrega datasets
    print("\nCarregando datasets...")
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )
    
    print(f"Dataset de treino: {len(train_dataset)} imagens")
    print(f"Dataset de validação: {len(val_dataset)} imagens")
    print(f"Classes: {train_dataset.classes}")
    
    # Modo de treinamento
    if args.mode == 'two_stage':
        print("\n=== MODO TWO-STAGE ===")
        print("Etapa 1: Extraindo features...")
        
        # DataLoaders para extração
        extract_loader_train = DataLoader(
            train_dataset, batch_size=args.batch_size * 2,
            shuffle=False, num_workers=args.num_workers
        )
        extract_loader_val = DataLoader(
            val_dataset, batch_size=args.batch_size * 2,
            shuffle=False, num_workers=args.num_workers
        )
        
        # Extrator de features
        feature_extractor = FeatureExtractorCNN(pretrained=True, freeze_layers=True)
        feature_extractor = feature_extractor.to(device)
        
        # Extrai features
        train_features, train_labels = extract_features(
            feature_extractor, extract_loader_train, device
        )
        val_features, val_labels = extract_features(
            feature_extractor, extract_loader_val, device
        )
        
        # Salva features
        np.save(os.path.join(save_dir, 'train_features.npy'), train_features)
        np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(save_dir, 'val_features.npy'), val_features)
        np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels)
        
        print("\nEtapa 2: Treinando MLP...")
        
        # Cria DataLoaders com features
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(train_features),
                torch.LongTensor(train_labels)
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(val_features),
                torch.LongTensor(val_labels)
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Modelo MLP apenas
        model = MLP_TB_Detector(
            input_size=2048,
            hidden_sizes=args.hidden_sizes,
            num_classes=2,
            dropout_rate=args.dropout
        )
        is_feature_mode = True
        
    else:  # end_to_end
        print("\n=== MODO END-TO-END ===")
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )
        
        # Modelo completo
        model = MLP_TB_Complete(
            use_pretrained=True,
            freeze_extractor=args.freeze_extractor,
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout
        )
        is_feature_mode = False
    
    model = model.to(device)
    
    # Imprime informações do modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParâmetros totais: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    
    # Loss e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Rastreador de métricas
    metrics = MetricsTracker()
    
    # Treinamento
    print(f"\nIniciando treinamento por {args.epochs} épocas...")
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Época {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Treina
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, is_feature_mode
        )
        
        # Valida
        val_loss, val_acc, val_auc, val_sens, val_spec = validate(
            model, val_loader, criterion, device, is_feature_mode
        )
        
        # Atualiza métricas
        metrics.update(
            epoch, train_loss, val_loss, train_acc, val_acc,
            val_auc, val_sens, val_spec
        )
        
        # Scheduler
        scheduler.step(val_auc)
        
        # Imprime resultados
        print(f"\nResultados da Época {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Val AUC-ROC: {val_auc:.4f}")
        print(f"  Sensitivity: {val_sens:.4f} | Specificity: {val_spec:.4f}")
        
        # Salva melhor modelo
        if val_auc > metrics.best_val_auc:
            print(f"  ✓ Novo melhor modelo! AUC: {val_auc:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, best_model_path)
    
    # Salva último modelo
    last_model_path = os.path.join(save_dir, 'last_model.pth')
    torch.save(model.state_dict(), last_model_path)
    
    # Plota métricas
    metrics.plot_metrics(os.path.join(save_dir, 'training_metrics.png'))
    
    # Salva histórico de métricas
    metrics_dict = {
        'train_losses': metrics.train_losses,
        'val_losses': metrics.val_losses,
        'train_accs': metrics.train_accs,
        'val_accs': metrics.val_accs,
        'val_aucs': metrics.val_aucs,
        'val_sensitivities': metrics.val_sensitivities,
        'val_specificities': metrics.val_specificities,
        'best_epoch': metrics.best_epoch,
        'best_val_auc': metrics.best_val_auc,
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"\n{'='*60}")
    print("TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}")
    print(f"Melhor época: {metrics.best_epoch + 1}")
    print(f"Melhor AUC-ROC: {metrics.best_val_auc:.4f}")
    print(f"Modelos salvos em: {save_dir}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Treinar MLP para Detecção de TB')
    
    # Dados
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Diretório com dados (deve conter train/ e val/)')
    parser.add_argument('--save-dir', type=str, default='../results',
                       help='Diretório para salvar resultados')
    
    # Modo de treinamento
    parser.add_argument('--mode', type=str, default='two_stage',
                       choices=['two_stage', 'end_to_end'],
                       help='Modo de treinamento')
    parser.add_argument('--freeze-extractor', action='store_true',
                       help='Congela feature extractor (apenas para end-to-end)')
    
    # Arquitetura
    parser.add_argument('--hidden-sizes', type=int, nargs='+', 
                       default=[512, 256, 128],
                       help='Tamanhos das camadas ocultas')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Taxa de dropout')
    
    # Treinamento
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Número de workers para DataLoader')
    
    args = parser.parse_args()
    
    # Treina
    train_mlp(args)


if __name__ == '__main__':
    main()
