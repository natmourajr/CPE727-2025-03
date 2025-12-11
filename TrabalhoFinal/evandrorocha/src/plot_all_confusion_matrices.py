"""
Script para visualizar as Matrizes de Confusão dos 4 Modelos lado a lado
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics(metrics_file):
    """Carrega a matriz de confusão e calcula métricas do arquivo JSON"""
    if not Path(metrics_file).exists():
        print(f"Aviso: Arquivo não encontrado: {metrics_file}")
        return None
        
    with open(metrics_file, 'r') as f:
        data = json.load(f)
        
    cm = np.array(data['confusion_matrix'])
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'cm': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc_roc': data.get('auc_roc', 0),
        'accuracy': data.get('accuracy', 0)
    }

def plot_all_cms(models_dict, save_path='results/all_confusion_matrices.png'):
    """
    Plota 4 matrizes de confusão em um grid 2x2
    
    Args:
        models_dict: Dict {Nome do Modelo: Caminho do arquivo JSON}
    """
    # Configurar estilo
    sns.set_style("white")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Ordem fixa para garantir consistência
    model_names = list(models_dict.keys())
    
    for i, (name, path) in enumerate(models_dict.items()):
        ax = axes[i]
        metrics = load_metrics(path)
        
        if metrics is None:
            ax.text(0.5, 0.5, "Dados não encontrados", ha='center', va='center')
            continue
            
        # Plotar Heatmap
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    cbar=False, annot_kws={"size": 16, "weight": "bold"},
                    xticklabels=['Normal', 'TB'],
                    yticklabels=['Normal', 'TB'])
        
        # Títulos e Labels
        ax.set_title(f"{name}", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Predito', fontsize=12)
        ax.set_ylabel('Real', fontsize=12)
        
        # Adicionar métricas abaixo da matriz
        metrics_text = (
            f"Sensibilidade: {metrics['sensitivity']*100:.1f}%\n"
            f"Especificidade: {metrics['specificity']*100:.1f}%\n"
            f"AUC-ROC: {metrics['auc_roc']*100:.2f}%"
        )
        
        # Caixa de texto com métricas
        ax.text(0.5, -0.25, metrics_text, 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle="round", alpha=0.1, facecolor='gray'))

    plt.suptitle("Comparação de Matrizes de Confusão: Classificação de Tuberculose", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Criar diretório se não existir
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo com sucesso em: {save_path}")

def main():
    # Definição dos modelos e arquivos
    models = {
        'Simple CNN (Baseline)': 'models/simplecnn_test_metrics.json',
        'ResNet-50 (Melhor Performance)': 'models/resnet50_test_metrics.json',
        'DenseNet-121 (Convergência Rápida)': 'models/densenet121_test_metrics.json',
        'EfficientNet-B0 (Melhor Especificidade)': 'models/efficientnet_b0_test_metrics.json'
    }
    
    print("Gerando visualização das matrizes de confusão...")
    plot_all_cms(models)

if __name__ == '__main__':
    main()
