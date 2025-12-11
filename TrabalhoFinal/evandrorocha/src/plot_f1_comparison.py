
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def main():
    models_dir = 'models'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Map model names to history files and display names
    models = {
        'simplecnn': {'file': 'simplecnn_history.json', 'label': 'CNN Baseline', 'color': '#d62728'}, # Red
        'resnet50': {'file': 'resnet50_history.json', 'label': 'ResNet-50', 'color': '#1f77b4'}, # Blue
        'densenet121': {'file': 'densenet121_history.json', 'label': 'DenseNet-121', 'color': '#ff7f0e'}, # Orange
        'efficientnet_b0': {'file': 'efficientnet_b0_history.json', 'label': 'EfficientNet-B0', 'color': '#2ca02c'} # Green
    }
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    max_epochs = 0
    
    for key, info in models.items():
        filepath = os.path.join(models_dir, info['file'])
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Skipping.")
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if 'val_f1' in data:
            f1_scores = data['val_f1']
            epochs = range(1, len(f1_scores) + 1)
            max_epochs = max(max_epochs, len(f1_scores))
            
            # Plot original (alpha) and smoothed
            # plt.plot(epochs, f1_scores, color=info['color'], alpha=0.3, linewidth=1)
            plt.plot(epochs, f1_scores, label=info['label'], color=info['color'], linewidth=2.5, marker='o', markersize=4)
            
            # Mark max point
            max_f1 = max(f1_scores)
            max_epoch = f1_scores.index(max_f1) + 1
            
            # Star marker
            plt.plot(max_epoch, max_f1, '*', color=info['color'], markersize=15, markeredgecolor='black', zorder=10)
            
            # Annotation
            label_text = f"Best: {max_f1:.2f}\n(Ep {max_epoch})"
            plt.annotate(label_text, 
                         xy=(max_epoch, max_f1), 
                         xytext=(10, 10), 
                         textcoords='offset points', 
                         fontsize=11, 
                         color=info['color'], 
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
            
    plt.title('Comparação de Desempenho: F1-Score (Validação)', fontsize=16, pad=20)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, max_epochs + 1)
    plt.ylim(0, 1.05)
    
    # Save
    output_path = os.path.join(results_dir, 'f1_score_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
