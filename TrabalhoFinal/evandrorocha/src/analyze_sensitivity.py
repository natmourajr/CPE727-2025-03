"""
Script para calcular e visualizar Sensibilidade e Especificidade
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_sensitivity_specificity(confusion_matrix):
    """
    Calcula Sensibilidade e Especificidade a partir da matriz de confusão
    
    Confusion Matrix:
    [[TN, FP],
     [FN, TP]]
    
    Args:
        confusion_matrix: Lista 2x2 com a matriz de confusão
        
    Returns:
        dict com sensitivity e specificity
    """
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    
    # Sensibilidade (Recall) = TP / (TP + FN)
    # Capacidade de detectar quem TEM tuberculose
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Especificidade = TN / (TN + FP)
    # Capacidade de identificar quem NÃO TEM tuberculose
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }


def plot_sensitivity_specificity(metrics_dict, save_path=None):
    """
    Cria gráfico de barras para Sensibilidade e Especificidade
    
    Args:
        metrics_dict: Dicionário com as métricas
        save_path: Caminho para salvar o gráfico (opcional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Sensibilidade e Especificidade
    metrics = ['Sensibilidade\n(Recall)', 'Especificidade']
    values = [metrics_dict['sensitivity'], metrics_dict['specificity']]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Sensibilidade vs Especificidade', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Meta: 90%')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Gráfico 2: Matriz de Confusão Detalhada
    cm = np.array([[metrics_dict['true_negatives'], metrics_dict['false_positives']],
                   [metrics_dict['false_negatives'], metrics_dict['true_positives']]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, 
                cbar_kws={'label': 'Contagem'},
                xticklabels=['Normal', 'TB'],
                yticklabels=['Normal', 'TB'])
    ax2.set_xlabel('Predito', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Real', fontsize=12, fontweight='bold')
    ax2.set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
    
    # Adicionar anotações explicativas
    total_normal = metrics_dict['true_negatives'] + metrics_dict['false_positives']
    total_tb = metrics_dict['false_negatives'] + metrics_dict['true_positives']
    
    spec_text = f"Especificidade: {metrics_dict['true_negatives']}/{total_normal} = {metrics_dict['specificity']*100:.1f}%"
    sens_text = f"Sensibilidade: {metrics_dict['true_positives']}/{total_tb} = {metrics_dict['sensitivity']*100:.1f}%"
    
    fig.text(0.5, 0.02, f"{spec_text}  |  {sens_text}", 
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Gráfico salvo em: {save_path}')
    
    return fig


def analyze_model_metrics(metrics_file):
    """
    Analisa métricas de um modelo e gera relatório
    
    Args:
        metrics_file: Caminho para o arquivo JSON com métricas
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Calcular sensibilidade e especificidade
    sens_spec = calculate_sensitivity_specificity(metrics['confusion_matrix'])
    
    # Combinar com métricas existentes
    full_metrics = {**metrics, **sens_spec}
    
    # Imprimir relatório
    print("="*60)
    print(f"ANÁLISE DE MÉTRICAS: {Path(metrics_file).stem}")
    print("="*60)
    print(f"\n📊 Métricas Gerais:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"  AUC-ROC:   {metrics['auc_roc']*100:.2f}%")
    
    print(f"\n🎯 Sensibilidade e Especificidade:")
    print(f"  Sensibilidade (Recall): {sens_spec['sensitivity']*100:.2f}%")
    print(f"    → De {sens_spec['true_positives'] + sens_spec['false_negatives']} pacientes com TB, detectou {sens_spec['true_positives']}")
    print(f"  Especificidade:         {sens_spec['specificity']*100:.2f}%")
    print(f"    → De {sens_spec['true_negatives'] + sens_spec['false_positives']} pacientes normais, identificou {sens_spec['true_negatives']}")
    
    print(f"\n📋 Matriz de Confusão:")
    print(f"  True Negatives (TN):  {sens_spec['true_negatives']}")
    print(f"  False Positives (FP): {sens_spec['false_positives']}")
    print(f"  False Negatives (FN): {sens_spec['false_negatives']}")
    print(f"  True Positives (TP):  {sens_spec['true_positives']}")
    
    print(f"\n💡 Interpretação Clínica:")
    if sens_spec['sensitivity'] >= 0.9:
        print(f"  ✅ Sensibilidade EXCELENTE ({sens_spec['sensitivity']*100:.1f}%)")
        print(f"     Modelo detecta a maioria dos casos de TB")
    elif sens_spec['sensitivity'] >= 0.8:
        print(f"  ✓ Sensibilidade BOA ({sens_spec['sensitivity']*100:.1f}%)")
    else:
        print(f"  ⚠️ Sensibilidade BAIXA ({sens_spec['sensitivity']*100:.1f}%)")
        print(f"     Muitos casos de TB podem passar despercebidos!")
    
    if sens_spec['specificity'] >= 0.9:
        print(f"  ✅ Especificidade EXCELENTE ({sens_spec['specificity']*100:.1f}%)")
        print(f"     Poucos falsos positivos")
    elif sens_spec['specificity'] >= 0.8:
        print(f"  ✓ Especificidade BOA ({sens_spec['specificity']*100:.1f}%)")
    else:
        print(f"  ⚠️ Especificidade BAIXA ({sens_spec['specificity']*100:.1f}%)")
        print(f"     Muitos pacientes normais diagnosticados incorretamente!")
    
    print("="*60)
    
    return full_metrics


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisar Sensibilidade e Especificidade')
    parser.add_argument('--metrics', type=str, required=True,
                        help='Caminho para o arquivo de métricas JSON')
    parser.add_argument('--plot', action='store_true',
                        help='Gerar gráfico')
    parser.add_argument('--save', type=str,
                        help='Caminho para salvar o gráfico')
    
    args = parser.parse_args()
    
    # Analisar métricas
    full_metrics = analyze_model_metrics(args.metrics)
    
    # Gerar gráfico se solicitado
    if args.plot:
        save_path = args.save if args.save else None
        plot_sensitivity_specificity(full_metrics, save_path)
        
        if not save_path:
            plt.show()


if __name__ == '__main__':
    main()
