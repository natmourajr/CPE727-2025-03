"""
Script de treinamento para modelos de detecção de tuberculose
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
from datetime import datetime
from typing import Dict, Tuple

from dataset import create_dataloaders
from models import create_model


class Trainer:
    """Classe para treinar modelos de detecção de tuberculose"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str = 'model',
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = './models'
    ):
        """
        Args:
            model: Modelo PyTorch
            model_name: Nome do modelo (para salvar arquivos)
            device: Dispositivo para treinamento ('cuda' ou 'cpu')
            learning_rate: Taxa de aprendizado
            weight_decay: Regularização L2
            save_dir: Diretório para salvar modelos
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Otimizador e loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=f'./runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        
        # Histórico
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Treina por uma época"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Valida o modelo"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calcular métricas
        val_loss = running_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='binary')
        val_recall = recall_score(all_labels, all_preds, average='binary')
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        val_auc = roc_auc_score(all_labels, all_probs)
        
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': val_f1,
            'auc_roc': val_auc,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """Loop de treinamento principal"""
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 60)
            
            # Treinar
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validar
            val_metrics = self.validate(val_loader)
            
            # Atualizar histórico
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/val', val_metrics['f1_score'], epoch)
            self.writer.add_scalar('AUC-ROC/val', val_metrics['auc_roc'], epoch)
            
            # Imprimir métricas
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f} | Val Acc: {val_metrics["accuracy"]:.4f}')
            print(f'Val Precision: {val_metrics["precision"]:.4f} | Val Recall: {val_metrics["recall"]:.4f}')
            print(f'Val F1: {val_metrics["f1_score"]:.4f} | Val AUC-ROC: {val_metrics["auc_roc"]:.4f}')
            
            # Learning rate scheduler
            self.scheduler.step(val_metrics['f1_score'])
            
            # Salvar melhor modelo
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                epochs_without_improvement = 0
                print(f'✓ Novo melhor modelo salvo! F1: {best_val_f1:.4f}')
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f'\nEarly stopping após {epoch+1} épocas')
                break
        
        self.writer.close()
        
        # Salvar histórico
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print('\nTreinamento concluído!')
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
        else:
            path = os.path.join(self.save_dir, f'{self.model_name}_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        print(f'Modelo salvo em: {path}')


def main():
    """Função principal para treinar o modelo"""
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Treinar modelo de detecção de tuberculose')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Nome do modelo (resnet50, efficientnet_b0, densenet121, simplecnn)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Número de épocas de treinamento')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data/shenzhen',
                        help='Diretório dos dados')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Tamanho da imagem (altura e largura)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Número de workers para DataLoader')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Diretório para salvar modelos')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Usando device: {device}')
    print(f'Modelo: {args.model}')
    print(f'Épocas: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    
    # Criar dataloaders
    print('\nCarregando dados...')
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Criar modelo
    print(f'\nCriando modelo: {args.model}')
    model = create_model(
        model_name=args.model,
        pretrained=True,
        num_classes=2,
        dropout=0.5
    )
    
    # Criar trainer
    trainer = Trainer(
        model=model,
        model_name=args.model,
        device=device,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
    
    # Treinar
    print('\nIniciando treinamento...')
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=10
    )
    
    # Avaliar no conjunto de teste
    print('\nAvaliando no conjunto de teste...')
    test_metrics = trainer.validate(test_loader)
    print(f'\nTest Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'Test Precision: {test_metrics["precision"]:.4f}')
    print(f'Test Recall: {test_metrics["recall"]:.4f}')
    print(f'Test F1: {test_metrics["f1_score"]:.4f}')
    print(f'Test AUC-ROC: {test_metrics["auc_roc"]:.4f}')
    
    # Salvar métricas de teste com nome do modelo
    test_metrics_path = os.path.join(args.save_dir, f'{args.model}_test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f'\nMétricas de teste salvas em: {test_metrics_path}')


if __name__ == '__main__':
    main()
