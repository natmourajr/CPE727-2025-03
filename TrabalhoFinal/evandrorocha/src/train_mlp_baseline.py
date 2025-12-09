"""
Script Simplificado para Treinar MLP Baseline com Features Manuais
Extrai features on-the-fly e treina MLP
"""

import os
import sys
import argparse
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import ndimage

import sys
import os

# Adicionar root e models ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'models'))

# Tentar importar MLP
try:
    from models.mlp import SimpleMLP
except ImportError:
    from mlp import SimpleMLP

from src.dataset import ShenzhenTBDataset


def extract_manual_features(image_path):
    """
    Extrai 81 features manuais de uma imagem
    
    Features:
    - 10 Intensidade
    - 16 Histograma
    - 20 GLCM (Textura)
    - 10 LBP (Textura Local)
    - 7 Momentos de Hu
    - 8 Gradiente
    - 10 FFT (Frequência)
    """
    # Carregar imagem em grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    
    features = []
    
    # 1. INTENSIDADE (10 features)
    features.extend([
        np.mean(img),
        np.std(img),
        np.var(img),
        np.min(img),
        np.max(img),
        np.median(img),
        np.percentile(img, 25),
        np.percentile(img, 75),
        float(np.mean((img - np.mean(img))**3) / (np.std(img)**3 + 1e-10)),  # skewness
        float(np.mean((img - np.mean(img))**4) / (np.std(img)**4 + 1e-10))   # kurtosis
    ])
    
    # 2. HISTOGRAMA (16 features)
    hist, _ = np.histogram(img, bins=16, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)
    features.extend(hist.tolist())
    
    # 3. GLCM - TEXTURA (20 features)
    # Normalizar imagem para 0-255
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
    
    # Calcular GLCM em 4 direções
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_norm, distances, angles, levels=256, symmetric=True, normed=True)
    
    # Propriedades GLCM
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        prop_values = graycoprops(glcm, prop)[0]
        features.extend(prop_values.tolist())
    
    # 4. LBP - TEXTURA LOCAL (10 features)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, n_points + 2))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-10)
    features.extend(lbp_hist.tolist())
    
    # 5. MOMENTOS DE HU (7 features)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    features.extend(hu_moments.tolist())
    
    # 6. GRADIENTE (8 features)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude),
        np.min(magnitude)
    ])
    
    # Laplaciano
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features.extend([
        np.mean(np.abs(laplacian)),
        np.std(laplacian),
        np.max(laplacian),
        np.min(laplacian)
    ])
    
    # 7. FFT - FREQUÊNCIA (10 features)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Dividir em regiões de frequência
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    
    # Baixa frequência (centro)
    low_freq = magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]
    features.extend([
        np.mean(low_freq),
        np.std(low_freq),
        np.max(low_freq)
    ])
    
    # Média frequência (anel ao redor do centro)
    mid_freq_outer = magnitude_spectrum[crow-60:crow+60, ccol-60:ccol+60]
    mid_freq_inner = magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]
    # Calcular média da região do anel (outer - inner)
    mid_freq_mean = (np.sum(mid_freq_outer) - np.sum(mid_freq_inner)) / (mid_freq_outer.size - mid_freq_inner.size + 1e-10)
    mid_freq_std = np.std(mid_freq_outer)
    mid_freq_max = np.max(mid_freq_outer)
    
    features.extend([
        mid_freq_mean,
        mid_freq_std,
        mid_freq_max
    ])
    
    # Alta frequência
    features.extend([
        np.mean(magnitude_spectrum),
        np.std(magnitude_spectrum),
        np.max(magnitude_spectrum),
        float(np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 95)))
    ])
    
    return np.array(features, dtype=np.float32)


def extract_features_from_dataset(data_dir, split='train'):
    """Extrai features de todo o dataset"""
    dataset = ShenzhenTBDataset(data_dir, mode=split, transform=None)
    
    all_features = []
    all_labels = []
    
    print(f"Extraindo features do split '{split}'...")
    for idx in tqdm(range(len(dataset))):
        img_path = dataset.image_paths[idx]
        label = dataset.labels[idx]
        
        try:
            features = extract_manual_features(img_path)
            all_features.append(features)
            all_labels.append(label)
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
            continue
    
    return np.array(all_features), np.array(all_labels)


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
    epoch_precision = precision_score(all_labels, all_preds)
    epoch_recall = recall_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, epoch_auc, epoch_precision, epoch_recall, epoch_f1


def main():
    parser = argparse.ArgumentParser(description='Treinar MLP Baseline com Features Manuais')
    parser.add_argument('--data-dir', type=str, default='./data/shenzhen',
                       help='Diretório do dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Criar diretório de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./results/mlp_baseline_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Extrair features
    print("\n" + "="*60)
    print("EXTRAÇÃO DE FEATURES MANUAIS")
    print("="*60)
    
    train_features, train_labels = extract_features_from_dataset(args.data_dir, 'train')
    val_features, val_labels = extract_features_from_dataset(args.data_dir, 'val')
    
    print(f"\nTrain: {train_features.shape}, Val: {val_features.shape}")
    print(f"Número de features: {train_features.shape[1]}")
    
    # Normalizar
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    
    # Salvar scaler para usar na avaliação
    import pickle
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler salvo em: {scaler_path}")
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels)),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_features), torch.LongTensor(val_labels)),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Modelo
    model = SimpleMLP(
        input_size=train_features.shape[1],
        hidden_size=128,
        num_classes=2,
        dropout_rate=0.3
    ).to(device)
    
    print(f"\nParâmetros do modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    # Treinamento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print("\n" + "="*60)
    print("TREINAMENTO")
    print("="*60)
    
    best_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    
    for epoch in range(args.epochs):
        print(f"\nÉpoca {epoch+1}/{args.epochs}")
        print("-"*60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f}")
        print(f"Val F1: {val_f1:.4f} | Val AUC-ROC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"✓ Melhor modelo salvo! AUC: {val_auc:.4f}")
    
    # Salvar histórico
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO!")
    print("="*60)
    print(f"Melhor AUC: {best_auc:.4f}")
    print(f"Resultados salvos em: {save_dir}")


if __name__ == '__main__':
    main()
