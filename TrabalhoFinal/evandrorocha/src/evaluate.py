"""
Script de avaliação e comparação de modelos
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)
import json
from typing import Dict, List

from dataset import create_dataloaders
from models import create_model


class ModelEvaluator:
    """Classe para avaliar e comparar modelos"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = {}
    
    def load_model(self, model_path: str, model_name: str = 'resnet50') -> torch.nn.Module:
        """Carrega um modelo treinado"""
        model = create_model(model_name=model_name, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        data_loader,
        model_name: str
    ) -> Dict:
        """Avalia um modelo"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calcular métricas
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)
        
        results = {
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc},
            'pr_curve': {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': pr_auc}
        }
        
        self.results[model_name] = results
        return results
    
    def plot_confusion_matrix(self, model_name: str, save_path: str = None):
        """Plota matriz de confusão"""
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'TB'],
            yticklabels=['Normal', 'TB']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, save_path: str = None):
        """Plota curvas ROC de todos os modelos"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            roc = results['roc_curve']
            plt.plot(
                roc['fpr'],
                roc['tpr'],
                label=f"{model_name} (AUC = {roc['auc']:.3f})"
            )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curves(self, save_path: str = None):
        """Plota curvas Precision-Recall"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            pr = results['pr_curve']
            plt.plot(
                pr['recall'],
                pr['precision'],
                label=f"{model_name} (AUC = {pr['auc']:.3f})"
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self) -> None:
        """Compara métricas de todos os modelos"""
        comparison = {}
        
        for model_name, results in self.results.items():
            report = results['classification_report']
            comparison[model_name] = {
                'Accuracy': report['accuracy'],
                'Precision': report['1']['precision'],
                'Recall': report['1']['recall'],
                'F1-Score': report['1']['f1-score'],
                'ROC-AUC': results['roc_curve']['auc'],
                'PR-AUC': results['pr_curve']['auc']
            }
        
        # Criar DataFrame para visualização
        import pandas as pd
        df = pd.DataFrame(comparison).T
        
        print("\n" + "="*80)
        print("COMPARAÇÃO DE MODELOS")
        print("="*80)
        print(df.to_string())
        print("="*80 + "\n")
        
        # Salvar comparação
        df.to_csv('./results/model_comparison.csv')
        
        # Plotar comparação
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics = df.columns
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            df[metric].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('./results/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Função principal para avaliar modelos"""
    
    # Configurações
    DATA_DIR = './data/shenzhen'
    BATCH_SIZE = 16
    IMAGE_SIZE = (224, 224)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Usando device: {device}')
    
    # Criar dataloader de teste
    _, _, test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=4
    )
    
    # Modelos para avaliar
    models_to_evaluate = {
        'Simple CNN': ('simple_cnn', './models/simple_cnn_best.pth'),
        'ResNet50': ('resnet50', './models/resnet50_best.pth'),
        'DenseNet121': ('densenet121', './models/densenet121_best.pth'),
        'EfficientNet-B0': ('efficientnet_b0', './models/efficientnet_best.pth'),
    }
    
    # Criar avaliador
    evaluator = ModelEvaluator(device=device)
    
    # Avaliar cada modelo
    for model_display_name, (model_name, model_path) in models_to_evaluate.items():
        if os.path.exists(model_path):
            print(f'\nAvaliando {model_display_name}...')
            model = evaluator.load_model(model_path, model_name)
            evaluator.evaluate_model(model, test_loader, model_display_name)
            evaluator.plot_confusion_matrix(
                model_display_name,
                save_path=f'./results/cm_{model_name}.png'
            )
        else:
            print(f'Modelo {model_display_name} não encontrado em {model_path}')
    
    # Comparar modelos
    if len(evaluator.results) > 0:
        print('\nComparando modelos...')
        evaluator.plot_roc_curves(save_path='./results/roc_comparison.png')
        evaluator.plot_pr_curves(save_path='./results/pr_comparison.png')
        evaluator.compare_models()


if __name__ == '__main__':
    main()
