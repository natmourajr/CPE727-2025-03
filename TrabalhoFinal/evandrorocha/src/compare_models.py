"""
Script para Comparar Modelos - Foco em Sensibilidade, Especificidade e AUC-ROC
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


def calculate_sensitivity_specificity(confusion_matrix):
    """
    Calcula Sensibilidade e Especificidade
    
    Confusion Matrix:
    [[TN, FP],
     [FN, TP]]
    """
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity


def load_model_metrics(metrics_file):
    """Carrega métricas de um modelo"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Calcular sensibilidade e especificidade
    sensitivity, specificity = calculate_sensitivity_specificity(
        metrics['confusion_matrix']
    )
    
    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],  # = sensitivity
        'f1_score': metrics['f1_score'],
        'auc_roc': metrics['auc_roc'],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': metrics['confusion_matrix']
    }


def compare_models(model_metrics_dict):
    """
    Compara múltiplos modelos
    
    Args:
        model_metrics_dict: Dict com {model_name: metrics_file_path}
    """
    results = {}
    
    print("="*80)
    print("COMPARAÇÃO DE MODELOS - DETECÇÃO DE TUBERCULOSE")
    print("="*80)
    
    # Carregar métricas de todos os modelos
    for model_name, metrics_file in model_metrics_dict.items():
        if Path(metrics_file).exists():
            results[model_name] = load_model_metrics(metrics_file)
            print(f"\n✅ {model_name}: Métricas carregadas")
        else:
            print(f"\n⚠️  {model_name}: Arquivo não encontrado - {metrics_file}")
    
    if not results:
        print("\n❌ Nenhum modelo encontrado!")
        return None
    
    # Criar tabela comparativa
    print("\n" + "="*80)
    print("📊 TABELA COMPARATIVA")
    print("="*80)
    
    # Cabeçalho
    print(f"\n{'Métrica':<25}", end='')
    for model_name in results.keys():
        print(f"{model_name:>15}", end='')
    print()
    print("-"*80)
    
    # Métricas principais
    metrics_to_compare = [
        ('Sensibilidade', 'sensitivity'),
        ('Especificidade', 'specificity'),
        ('AUC-ROC', 'auc_roc'),
        ('Acurácia', 'accuracy'),
        ('Precisão', 'precision'),
        ('F1-Score', 'f1_score'),
    ]
    
    for metric_label, metric_key in metrics_to_compare:
        print(f"{metric_label:<25}", end='')
        values = []
        for model_name in results.keys():
            value = results[model_name][metric_key]
            values.append(value)
            print(f"{value*100:>14.2f}%", end='')
        print()
        
        # Destacar melhor modelo
        best_idx = np.argmax(values)
        best_model = list(results.keys())[best_idx]
        if metric_label in ['Sensibilidade', 'Especificidade', 'AUC-ROC']:
            print(f"{'':25}{'→ Melhor: ' + best_model:>15}")
    
    print("="*80)
    
    # Análise detalhada das métricas principais
    print("\n🎯 ANÁLISE DAS MÉTRICAS PRINCIPAIS")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Sensibilidade: {metrics['sensitivity']*100:.2f}% (detecta {metrics['sensitivity']*100:.0f}% dos casos de TB)")
        print(f"  Especificidade: {metrics['specificity']*100:.2f}% (identifica {metrics['specificity']*100:.0f}% dos normais)")
        print(f"  AUC-ROC: {metrics['auc_roc']*100:.2f}% (capacidade discriminativa)")
        
        # Matriz de confusão
        cm = metrics['confusion_matrix']
        tn, fp = cm[0]
        fn, tp = cm[1]
        print(f"  Matriz: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Recomendação
    print("\n" + "="*80)
    print("💡 RECOMENDAÇÃO")
    print("="*80)
    
    # Encontrar melhor modelo por métrica
    best_sensitivity = max(results.items(), key=lambda x: x[1]['sensitivity'])
    best_specificity = max(results.items(), key=lambda x: x[1]['specificity'])
    best_auc = max(results.items(), key=lambda x: x[1]['auc_roc'])
    
    print(f"\n🏆 Melhor Sensibilidade: {best_sensitivity[0]} ({best_sensitivity[1]['sensitivity']*100:.2f}%)")
    print(f"🏆 Melhor Especificidade: {best_specificity[0]} ({best_specificity[1]['specificity']*100:.2f}%)")
    print(f"🏆 Melhor AUC-ROC: {best_auc[0]} ({best_auc[1]['auc_roc']*100:.2f}%)")
    
    # Modelo recomendado (melhor AUC-ROC)
    print(f"\n✅ MODELO RECOMENDADO: {best_auc[0]}")
    print(f"   Razão: Melhor AUC-ROC ({best_auc[1]['auc_roc']*100:.2f}%) indica")
    print(f"   superior capacidade discriminativa entre TB e Normal")
    
    return results


def plot_comparison(results, save_dir='results'):
    """Gera gráficos comparativos"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Gráfico de barras: Sensibilidade, Especificidade, AUC-ROC
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    models = list(results.keys())
    metrics = ['Sensibilidade', 'Especificidade', 'AUC-ROC']
    
    x = np.arange(len(models))
    width = 0.25
    
    sensitivity_values = [results[m]['sensitivity']*100 for m in models]
    specificity_values = [results[m]['specificity']*100 for m in models]
    auc_values = [results[m]['auc_roc']*100 for m in models]
    
    bars1 = ax.bar(x - width, sensitivity_values, width, label='Sensibilidade', color='#2ecc71')
    bars2 = ax.bar(x, specificity_values, width, label='Especificidade', color='#3498db')
    bars3 = ax.bar(x + width, auc_values, width, label='AUC-ROC', color='#e74c3c')
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Modelos: Métricas Principais', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = f'{save_dir}/model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n📊 Gráfico comparativo salvo em: {save_path}')
    
    # 2. Tabela resumo
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Modelo', 'Sensibilidade', 'Especificidade', 'AUC-ROC', 'Acurácia'])
    
    for model_name, metrics in results.items():
        table_data.append([
            model_name,
            f"{metrics['sensitivity']*100:.2f}%",
            f"{metrics['specificity']*100:.2f}%",
            f"{metrics['auc_roc']*100:.2f}%",
            f"{metrics['accuracy']*100:.2f}%"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.2, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilizar cabeçalho
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Destacar melhor valor em cada coluna
    for col_idx, metric_key in enumerate(['sensitivity', 'specificity', 'auc_roc', 'accuracy'], 1):
        values = [results[m][metric_key] for m in results.keys()]
        best_idx = np.argmax(values)
        table[(best_idx + 1, col_idx)].set_facecolor('#2ecc71')
        table[(best_idx + 1, col_idx)].set_text_props(weight='bold')
    
    plt.title('Tabela Comparativa de Modelos', fontsize=14, fontweight='bold', pad=20)
    
    save_path = f'{save_dir}/model_comparison_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'📊 Tabela comparativa salva em: {save_path}')


def main():
    """Função principal"""
    # Definir modelos a comparar
    models = {
        #'MLP Baseline': 'models/mlp_baseline_test_metrics.json',
        'Simple CNN': 'models/simplecnn_test_metrics.json',
        'ResNet50': 'models/resnet50_test_metrics.json',
        'DenseNet121': 'models/densenet121_test_metrics.json',
        'EfficientNet-B0': 'models/efficientnet_b0_test_metrics.json',
    }
    
    # Comparar modelos
    results = compare_models(models)
    
    if results:
        # Gerar gráficos
        plot_comparison(results, save_dir='results')
        
        print("\n✅ Comparação concluída!")
        print("📁 Resultados salvos em: results/")


if __name__ == '__main__':
    main()
