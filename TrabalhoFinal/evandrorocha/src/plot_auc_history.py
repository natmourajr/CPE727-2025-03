
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_auc_data(log_dir):
    """Extracts AUC-ROC/val steps and values from a TensorBoard log directory."""
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # Check available tags
        tags = ea.Tags()['scalars']
        target_tag = 'AUC-ROC/val'
        
        if target_tag in tags:
            events = ea.Scalars(target_tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            return steps, values
        else:
            print(f"Tag {target_tag} not found in {log_dir}")
            return [], []
    except Exception as e:
        print(f"Error reading {log_dir}: {e}")
        return [], []

def main():
    runs_dir = 'runs'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Map model keys to specific log directories found in runs/
    # Using the exact folder names identified in previous steps
    models = {
        'SimpleCNN': {
            'path': os.path.join(runs_dir, 'simplecnn_20251208-232358'), # Using the non-empty run
            'color': '#d62728', # Red
            'label': 'CNN Baseline'
        },
        'ResNet50': {
            'path': os.path.join(runs_dir, 'resnet50_20251208-184711'),
            'color': '#1f77b4', # Blue
            'label': 'ResNet-50'
        },
        'DenseNet121': {
            'path': os.path.join(runs_dir, 'densenet121_20251208-200355'),
            'color': '#ff7f0e', # Orange
            'label': 'DenseNet-121'
        },
        'EfficientNetB0': {
            'path': os.path.join(runs_dir, 'efficientnet_b0_20251208-210627'),
            'color': '#2ca02c', # Green
            'label': 'EfficientNet-B0'
        }
    }
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    max_epochs = 0
    
    for key, info in models.items():
        if not os.path.exists(info['path']):
            print(f"Warning: Path {info['path']} does not exist. Skipping.")
            continue
            
        steps, values = get_auc_data(info['path'])
        
        if not steps:
            print(f"Warning: No valid AUC data found for {key} in {info['path']}")
            continue

        max_epochs = max(max_epochs, max(steps))
        
        # Find max point first to include in label
        max_val = max(values)
        max_step = steps[values.index(max_val)]
        
        # Create label with stats
        label_text = f"{info['label']} (Best: {max_val:.3f}, Ep {max_step})"
            
        # Plot curve
        plt.plot(steps, values, label=label_text, color=info['color'], linewidth=2.5, marker='o', markersize=4)
        
        # Add star marker for best epoch (no extra label)
        plt.plot(max_step, max_val, '*', color=info['color'], markersize=15, markeredgecolor='black', zorder=10)
            
    plt.title('Comparação de Desempenho: AUC-ROC por Época', fontsize=16, pad=20)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('AUC-ROC (Validação)', fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, max_epochs + 2)
    plt.ylim(0.4, 1.0) # AUC usually between 0.5 and 1.0
    
    output_path = os.path.join(results_dir, 'auc_per_epoch_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
