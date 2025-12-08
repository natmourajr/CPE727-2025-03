"""
Script de Treinamento MLP com Features Manuais
Treina MLP usando features extraídas manualmente (GLCM, LBP, etc.)
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import SimpleMLP, MLP_TB_Detector


def load_manual_features(features_dir, split='train'):
    """
    Carrega features manuais extraídas
    
    Args:
        features_dir: diretório com features
        split: 'train', 'val', ou 'test'
    
    Returns:
        features, labels
    """
    features_path = os.path.join(features_dir, f'{split}_features_manual.npy')
    labels_path = os.path.join(features_dir, f'{split}_labels_manual.npy')
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features não encontradas: {features_path}")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print(f"Carregadas {len(features)} amostras do split '{split}'")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return features, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Treina por uma época"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
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
        for features, labels in tqdm(dataloader, desc="Validating"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
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


def main():
    parser = argparse.ArgumentParser(
        description='Treinar MLP com Features Manuais'
    )
    
    parser.add_argument('--features-dir', type=str, required=True,
                       help='Diretório com features extraídas')
    parser.add_argument('--save-dir', type=str, default='../results',
                       help='Diretório para salvar resultados')
    
    # Arquitetura
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'deep'],
                       help='Tipo de MLP (simple ou deep)')
    parser.add_argument('--hidden-sizes', type=int, nargs='+',
                       default=[128, 64],
                       help='Tamanhos das camadas ocultas (apenas para deep)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Taxa de dropout')
    
    # Treinamento
    parser.add_argument('--epochs', type=int, default=200,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Pré-processamento
    parser.add_argument('--normalize', action='store_true',
                       help='Normaliza features (StandardScaler)')
    
    args = parser.parse_args()
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Diretório de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"mlp_manual_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Salva argumentos
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Carrega features
    print("\nCarregando features...")
    train_features, train_labels = load_manual_features(args.features_dir, 'train')
    val_features, val_labels = load_manual_features(args.features_dir, 'val')
    
    # Normalização
    if args.normalize:
        print("\nNormalizando features...")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        
        # Salva scaler
        import pickle
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler salvo para uso futuro")
    
    # Número de features
    num_features = train_features.shape[1]
    print(f"\nNúmero de features: {num_features}")
    
    # Cria DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(train_features),
            torch.LongTensor(train_labels)
        ),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(val_features),
            torch.LongTensor(val_labels)
        ),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Cria modelo
    if args.model_type == 'simple':
        model = SimpleMLP(
            input_size=num_features,
            hidden_size=128,
            num_classes=2,
            dropout_rate=args.dropout
        )
    else:  # deep
        model = MLP_TB_Detector(
            input_size=num_features,
            hidden_sizes=args.hidden_sizes,
            num_classes=2,
            dropout_rate=args.dropout
        )
    
    model = model.to(device)
    
    print(f"\nModelo: {args.model_type}")
    print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
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
            }, os.path.join(save_dir, 'best_model.pth'))
    
    # Salva histórico
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plota métricas
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['val_auc'])
    axes[1, 0].set_title('Validation AUC-ROC')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['val_sens'], label='Sensitivity')
    axes[1, 1].plot(history['val_spec'], label='Specificity')
    axes[1, 1].set_title('Sensitivity & Specificity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300)
    
    print(f"\n{'='*60}")
    print("TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}")
    print(f"Melhor AUC: {best_auc:.4f}")
    print(f"Resultados salvos em: {save_dir}")


if __name__ == '__main__':
    main()
