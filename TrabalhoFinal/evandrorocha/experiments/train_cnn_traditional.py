"""
Script de Treinamento para SimpleCNN_TB
CNN Tradicional Otimizada para Detecção de Tuberculose
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.traditional_cnn import SimpleCNN_TB, get_traditional_cnn


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Treina por uma época"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Valida o modelo"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return epoch_loss, epoch_acc, epoch_auc, sensitivity, specificity


def plot_training_history(history, save_path):
    """Plota histórico de treinamento"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[1, 0].plot(history['val_auc'], marker='d', color='green')
    axes[1, 0].axhline(y=max(history['val_auc']), color='r', linestyle='--',
                      label=f'Best: {max(history["val_auc"]):.4f}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Validation AUC-ROC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Sensitivity & Specificity
    axes[1, 1].plot(history['val_sens'], label='Sensitivity', marker='^')
    axes[1, 1].plot(history['val_spec'], label='Specificity', marker='v')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Sensitivity and Specificity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Treinar SimpleCNN_TB para Detecção de TB'
    )
    
    # Dados
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Diretório com dados (deve conter train/ e val/)')
    parser.add_argument('--save-dir', type=str, default='../results',
                       help='Diretório para salvar resultados')
    
    # Modelo
    parser.add_argument('--model-type', type=str, default='simple_tb',
                       choices=['simple_tb', 'simple', 'traditional'],
                       help='Tipo de CNN')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Taxa de dropout')
    
    # Treinamento
    parser.add_argument('--epochs', type=int, default=150,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Número de workers')
    
    args = parser.parse_args()
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Diretório de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"cnn_{args.model_type}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Salva argumentos
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Data Augmentation FORTE (essencial para dataset pequeno)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
    
    # Datasets
    print("\nCarregando datasets...")
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )
    
    print(f"Train: {len(train_dataset)} imagens")
    print(f"Val: {len(val_dataset)} imagens")
    print(f"Classes: {train_dataset.classes}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    
    # Modelo
    print(f"\nCriando modelo: {args.model_type}")
    model = get_traditional_cnn(
        model_type=args.model_type,
        num_classes=len(train_dataset.classes),
        dropout_rate=args.dropout
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros totais: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    print(f"Tamanho estimado: ~{total_params * 4 / (1024**2):.2f} MB")
    
    # Loss e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Treinamento
    print(f"\nIniciando treinamento por {args.epochs} épocas...")
    
    best_auc = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc': [], 'val_sens': [], 'val_spec': []
    }
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Época {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Treina
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Valida
        val_loss, val_acc, val_auc, val_sens, val_spec = validate(
            model, val_loader, criterion, device
        )
        
        # Atualiza histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_sens'].append(val_sens)
        history['val_spec'].append(val_spec)
        
        # Scheduler
        scheduler.step(val_auc)
        
        # Imprime
        print(f"\nResultados:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        print(f"  Sensitivity: {val_sens:.4f} | Specificity: {val_spec:.4f}")
        
        # Salva melhor modelo
        if val_auc > best_auc:
            best_auc = val_auc
            print(f"  ✓ Novo melhor modelo! AUC: {val_auc:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
    
    # Salva último modelo
    torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
    
    # Salva histórico
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plota
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    
    print(f"\n{'='*60}")
    print("TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}")
    print(f"Melhor AUC: {best_auc:.4f}")
    print(f"Resultados salvos em: {save_dir}")


if __name__ == '__main__':
    main()
