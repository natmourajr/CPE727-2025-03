
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def scan_runs(root_dir='./runs'):
    runs = {}
    for dirname in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dirname)
        if not os.path.isdir(dir_path):
            continue
            
        # Initialize EventAccumulator
        try:
            ea = EventAccumulator(dir_path)
            ea.Reload()
            
            # Check for scaler tags
            tags = ea.Tags()['scalars']
            if 'AUC-ROC/val' in tags:
                # Extract data
                auc_data = ea.Scalars('AUC-ROC/val')
                steps = [x.step for x in auc_data]
                values = [x.value for x in auc_data]
                
                # Identify model from folder name or assumption
                model_name = dirname
                if 'resnet' in dirname.lower():
                    model_name = 'ResNet-50'
                elif 'densenet' in dirname.lower():
                    model_name = 'DenseNet-121'
                elif 'efficientnet' in dirname.lower():
                    model_name = 'EfficientNet-B0'
                elif 'cnn' in dirname.lower() or 'baseline' in dirname.lower():
                    model_name = 'CNN Baseline'
                
                runs[model_name] = {'steps': steps, 'values': values, 'dir': dirname}
                print(f"Found {len(values)} steps in {dirname} (Likely: {model_name})")
                
                # If identifier not in name, try to guess by max accuracy or other heuristics if needed
                # But for now let's just see what we get
        except Exception as e:
            print(f"Error reading {dirname}: {e}")
            
    return runs

if __name__ == "__main__":
    found_runs = scan_runs()
    
    # Simple logic to disambiguate if multiple runs map to same name 
    # (picking the one with most epochs or latest timestamp)
    print("\nSummary of extracted runs:")
    for name, data in found_runs.items():
        print(f"{name}: {len(data['values'])} epochs. Final AUC: {data['values'][-1]:.4f}")
