"""
Funções utilitárias para o projeto
"""
import os
import torch
import numpy as np
import random
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List


def set_seed(seed: int = 42):
    """Define seed para reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Retorna o dispositivo disponível"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU disponível: {torch.cuda.get_device_name(0)}")
        print(f"   Memória total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = 'cpu'
        print("⚠️  Usando CPU (treinamento será mais lento)")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Conta o número de parâmetros treináveis do modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: dict, save_path: str):
    """Salva configuração em JSON"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(config_path: str) -> dict:
    """Carrega configuração de JSON"""
    with open(config_path, 'r') as f:
        return json.load(f)


def plot_training_history(history: dict, save_path: str = None):
    """Plota histórico de treinamento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_experiment_name(model_name: str) -> str:
    """Cria nome único para experimento"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def log_metrics(metrics: dict, log_file: str):
    """Loga métricas em arquivo"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{timestamp}]\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


class EarlyStopping:
    """Early stopping para interromper treinamento"""
    
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 0.0):
        """
        Args:
            patience: Número de épocas sem melhoria antes de parar
            mode: 'max' para métricas que devem aumentar, 'min' para diminuir
            delta: Melhoria mínima considerável
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Verifica se deve parar o treinamento
        
        Returns:
            True se deve continuar, False se deve parar
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def format_time(seconds: float) -> str:
    """Formata tempo em segundos para string legível"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_model_summary(model: torch.nn.Module):
    """Imprime resumo do modelo"""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*60 + "\n")
