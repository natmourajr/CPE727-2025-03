"""
Visualization utilities for training results

All plots are saved to disk, not displayed (as per requirements).
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json


class ResultsVisualizer:
    """
    Generate and save training plots
    
    Args:
        output_dir (str): Base directory for saving results
    """
    
    def __init__(self, output_dir='results'):
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📁 Saving results to: {self.save_dir}")
    
    def plot_training_curves(self, history, model_name):
        """
        Plot training and validation loss/accuracy curves
        
        Args:
            history (dict): Training history with loss and accuracy
            model_name (str): Name of the model for the title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        
        # Mark best epoch if available
        if history.get('best_epoch') is not None:
            best_epoch = history['best_epoch']
            ax1.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1.5, 
                       label=f'Best Model (epoch {best_epoch})', alpha=0.7)
        
        # Mark early stopping if it occurred
        if history.get('early_stopped', False) and history.get('stopped_epoch') is not None:
            stopped_epoch = history['stopped_epoch']
            ax1.axvline(x=stopped_epoch, color='orange', linestyle=':', linewidth=1.5,
                       label=f'Early Stop (epoch {stopped_epoch})', alpha=0.7)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        
        # Mark best epoch if available
        if history.get('best_epoch') is not None:
            best_epoch = history['best_epoch']
            ax2.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1.5,
                       label=f'Best Model (epoch {best_epoch})', alpha=0.7)
        
        # Mark early stopping if it occurred
        if history.get('early_stopped', False) and history.get('stopped_epoch') is not None:
            stopped_epoch = history['stopped_epoch']
            ax2.axvline(x=stopped_epoch, color='orange', linestyle=':', linewidth=1.5,
                       label=f'Early Stop (epoch {stopped_epoch})', alpha=0.7)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'{model_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {model_name}_training_curves.png")
    
    def plot_comparison(self, results_dict):
        """
        Compare multiple models' performance
        
        Args:
            results_dict (dict): Dictionary mapping model names to their training histories
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        for model_name, history in results_dict.items():
            epochs = range(1, len(history['val_loss']) + 1)
            
            # Validation loss
            ax1.plot(epochs, history['val_loss'], label=model_name, linewidth=2)
            
            # Validation accuracy
            ax2.plot(epochs, history['val_acc'], label=model_name, linewidth=2)
            
            # Training loss
            ax3.plot(epochs, history['train_loss'], label=model_name, linewidth=2)
            
            # Training accuracy
            ax4.plot(epochs, history['train_acc'], label=model_name, linewidth=2)
        
        # Configure subplots
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Validation Loss', fontsize=12)
        ax1.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Training Loss', fontsize=12)
        ax3.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Training Accuracy (%)', fontsize=12)
        ax4.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: model_comparison.png")
    
    def save_summary(self, summary_data):
        """
        Save experiment summary to JSON
        
        Args:
            summary_data (dict): Summary data to save
        """
        summary_path = os.path.join(self.save_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"  ✓ Saved: summary.json")
    
    def get_save_dir(self):
        """Get the current save directory path"""
        return self.save_dir

