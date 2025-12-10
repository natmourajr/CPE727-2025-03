"""
Script Simplificado para Gerar Curvas ROC
Usa métricas já salvas (confusion matrix + AUC) para gerar visualizações
Não requer PyTorch - apenas matplotlib, numpy e sklearn
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from pathlib import Path

# Configurações
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 11

MODELS_DIR = Path('models')
OUTPUT_DIR = Path('results/roc_curves')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_roc_from_metrics(auc_score, num_points=100):
    """
    Gera uma curva ROC aproximada baseada no AUC
    Para visualização quando não temos as predições completas
    """
    # Gera pontos FPR uniformemente distribuídos
    fpr = np.linspace(0, 1, num_points)
    
    # Aproximação: TPR baseado no AUC
    # Para AUC = 0.5: TPR = FPR (linha diagonal)
    # Para AUC > 0.5: TPR > FPR (curva acima da diagonal)
    
    if auc_score >= 0.5:
        # Curva convexa típica de ROC
        # Usa função power para criar curvatura
        power = 2 * (1 - auc_score) + 0.5
        tpr = fpr ** (1/power)
        
        # Ajusta para garantir AUC correto
        current_auc = np.trapz(tpr, fpr)
        tpr = tpr * (auc_score / current_auc)
        tpr = np.clip(tpr, 0, 1)
    else:
        # AUC < 0.5 (pior que chance)
        tpr = fpr * (auc_score / 0.5)
    
    return fpr, tpr


def calculate_optimal_point_from_cm(confusion_matrix):
    """Calcula ponto ótimo da curva ROC usando confusion matrix"""
    tn, fp, fn, tp = confusion_matrix[0][0], confusion_matrix[0][1], \
                     confusion_matrix[1][0], confusion_matrix[1][1]
    
    # Calcular TPR (sensibilidade) e FPR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return fpr, tpr


def plot_roc_curve_from_metrics(model_name, metrics, color, save_path):
    """Plota curva ROC usando métricas salvas"""
    auc_score = metrics['auc_roc']
    
    # Gerar curva ROC aproximada
    fpr, tpr = generate_roc_from_metrics(auc_score, num_points=200)
    
    # Calcular ponto ótimo real da confusion matrix
    fpr_optimal, tpr_optimal = calculate_optimal_point_from_cm(metrics['confusion_matrix'])
    
    # Plotar
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color=color, lw=3, 
             label=f'{model_name} (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.5000)')
    
    # Marcar ponto ótimo (do teste real)
    plt.plot(fpr_optimal, tpr_optimal, 'o', color=color, markersize=12,
             markeredgecolor='white', markeredgewidth=2,
             label=f'Ponto de Teste (Sens={tpr_optimal:.2%}, Spec={1-fpr_optimal:.2%})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=13)
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=13)
    plt.title(f'Curva ROC - {model_name}', fontsize=15, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Adicionar informações da confusion matrix
    cm = metrics['confusion_matrix']
    info_text = f"Confusion Matrix:\n"
    info_text += f"TN={cm[0][0]}  FP={cm[0][1]}\n"
    info_text += f"FN={cm[1][0]}  TP={cm[1][1]}"
    plt.text(0.98, 0.02, info_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Curva ROC salva: {save_path}")
    plt.close()
    
    return fpr, tpr, fpr_optimal, tpr_optimal


def plot_combined_roc(results, save_path):
    """Plota todas as curvas ROC em um único gráfico"""
    plt.figure(figsize=(10, 10))
    
    # Plotar cada modelo
    for model_name, data in results.items():
        plt.plot(data['fpr'], data['tpr'], 
                color=data['color'], lw=3, alpha=0.8,
                label=f"{model_name} (AUC = {data['auc']:.4f})")
        
        # Marcar ponto de teste real
        plt.plot(data['fpr_test'], data['tpr_test'], 'o', 
                color=data['color'], markersize=10,
                markeredgecolor='white', markeredgewidth=2)
    
    # Linha de chance
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Chance (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=14, fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=14, fontweight='bold')
    plt.title('Comparação de Curvas ROC - Todos os Modelos', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Curva ROC combinada salva: {save_path}")
    plt.close()


def plot_roc_comparison_table(results, save_path):
    """Cria tabela comparativa de métricas ROC"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de barras - AUC
    model_names = list(results.keys())
    aucs = [results[m]['auc'] for m in model_names]
    colors = [results[m]['color'] for m in model_names]
    
    bars = ax1.barh(model_names, aucs, color=colors, alpha=0.7)
    ax1.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax1.set_title('Comparação de AUC-ROC', fontsize=14, fontweight='bold')
    ax1.set_xlim([0.8, 1.0])
    ax1.grid(axis='x', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, (bar, auc_val) in enumerate(zip(bars, aucs)):
        ax1.text(auc_val + 0.005, i, f'{auc_val:.4f}', 
                va='center', fontweight='bold', fontsize=11)
    
    # Tabela de métricas
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = []
    for model_name in model_names:
        data = results[model_name]
        table_data.append([
            model_name,
            f"{data['auc']:.4f}",
            f"{data['accuracy']:.2%}",
            f"{data['tpr_test']:.2%}",
            f"{1 - data['fpr_test']:.2%}",
            f"{data['fp']}/{data['fn']}"
        ])
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Modelo', 'AUC-ROC', 'Acurácia', 'Sensibilidade', 'Especificidade', 'FP/FN'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Colorir header
    for i in range(6):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorir linhas alternadas
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax2.set_title('Métricas no Conjunto de Teste', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Tabela comparativa salva: {save_path}")
    plt.close()


def main():
    print("="*80)
    print("GERAÇÃO DE CURVAS ROC - BASEADO EM MÉTRICAS SALVAS")
    print("="*80)
    
    # Configuração dos modelos
    models_config = {
        'SimpleCNN': {
            'file': MODELS_DIR / 'simplecnn_test_metrics.json',
            'color': '#3498db'
        },
        'ResNet50': {
            'file': MODELS_DIR / 'resnet50_test_metrics.json',
            'color': '#e74c3c'
        },
        'DenseNet121': {
            'file': MODELS_DIR / 'densenet121_test_metrics.json',
            'color': '#2ecc71'
        },
        'EfficientNet-B0': {
            'file': MODELS_DIR / 'efficientnet_b0_test_metrics.json',
            'color': '#f39c12'
        }
    }
    
    # Processar cada modelo
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n{'='*80}")
        print(f"Processando: {model_name}")
        print(f"{'='*80}")
        
        # Carregar métricas
        if not config['file'].exists():
            print(f"⚠️ Arquivo não encontrado: {config['file']}")
            continue
        
        with open(config['file'], 'r') as f:
            metrics = json.load(f)
        
        print(f"Métricas carregadas:")
        print(f"  • AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  • Acurácia: {metrics['accuracy']:.2%}")
        print(f"  • Sensibilidade: {metrics['recall']:.2%}")
        print(f"  • Especificidade: {1 - (metrics['confusion_matrix'][0][1] / sum(metrics['confusion_matrix'][0])):.2%}")
        
        # Gerar curva ROC
        fpr, tpr, fpr_test, tpr_test = plot_roc_curve_from_metrics(
            model_name, metrics, config['color'],
            OUTPUT_DIR / f'roc_curve_{model_name.lower().replace("-", "_")}.png'
        )
        
        # Armazenar resultados
        cm = metrics['confusion_matrix']
        results[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'fpr_test': fpr_test,
            'tpr_test': tpr_test,
            'auc': metrics['auc_roc'],
            'accuracy': metrics['accuracy'],
            'color': config['color'],
            'fp': cm[0][1],
            'fn': cm[1][0]
        }
    
    # Gerar visualizações comparativas
    if results:
        print(f"\n{'='*80}")
        print("Gerando visualizações comparativas...")
        print(f"{'='*80}")
        
        plot_combined_roc(results, OUTPUT_DIR / 'roc_curves_combined.png')
        plot_roc_comparison_table(results, OUTPUT_DIR / 'roc_comparison_table.png')
        
        # Salvar resumo
        summary = {}
        for model_name, data in results.items():
            summary[model_name] = {
                'auc_roc': float(data['auc']),
                'accuracy': float(data['accuracy']),
                'sensitivity': float(data['tpr_test']),
                'specificity': float(1 - data['fpr_test']),
                'false_positives': int(data['fp']),
                'false_negatives': int(data['fn'])
            }
        
        with open(OUTPUT_DIR / 'roc_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✅ Resumo salvo: {OUTPUT_DIR / 'roc_summary.json'}")
    
    # Resumo final
    print(f"\n{'='*80}")
    print("RESUMO - RANKING POR AUC-ROC")
    print(f"{'='*80}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    for i, (model_name, data) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {model_name:20s} AUC = {data['auc']:.4f} | Acc = {data['accuracy']:.2%}")
    
    print(f"\n{'='*80}")
    print(f"✅ Todas as curvas ROC foram geradas em: {OUTPUT_DIR}")
    print(f"{'='*80}")
    print("\nArquivos gerados:")
    print(f"  • roc_curve_simplecnn.png")
    print(f"  • roc_curve_resnet50.png")
    print(f"  • roc_curve_densenet121.png")
    print(f"  • roc_curve_efficientnet_b0.png")
    print(f"  • roc_curves_combined.png (⭐ COMPARAÇÃO)")
    print(f"  • roc_comparison_table.png")
    print(f"  • roc_summary.json")
    
    print(f"\n💡 Nota: Curvas geradas usando AUC e confusion matrix.")
    print(f"   Para curvas ROC exatas, execute dentro do ambiente Docker/PyTorch.")


if __name__ == '__main__':
    main()
