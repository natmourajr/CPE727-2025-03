"""
Main script for training models on image datasets

Phase 1.1: Training with Early Stopping
- Train MLP, CNN, and ResNet-18 on FashionMNIST dataset
- Implement early stopping following textbook algorithm
- Generate training curves and comparison plots
- Save results to timestamped directory
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data.fashion_mnist_loader import FashionMNISTDataModule
from models.mlp import MLP
from models.cnn import SimpleCNN
from models.resnet import ResNet18CIFAR
from training.trainer import Trainer
from training.early_stopping import EarlyStopping
from visualization.plots import ResultsVisualizer
import json
import os


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"🌱 Random seed set to: {seed}")


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def main():
    """Main training loop"""
    
    print_header("Model Training - Phase 1.1")
    print("Training MLP and CNN models with Early Stopping")
    print("Following textbook algorithm from Goodfellow et al.")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device configuration - prioritize CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🖥️  Using device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Using device: CPU")
        print("⚠️  Training on CPU - this may take a while!")
    
    # ========================================
    # Data Loading
    # ========================================
    
    data_module = FashionMNISTDataModule(
        data_dir='./data',
        batch_size=64,
        val_split=0.1,
        num_workers=2,
        data_cap_rate=20  # Use 1/20 of data for overfitting experiments
    )
    
    print_header(f"Loading {data_module.name} Dataset")
    data_module.prepare_data()
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print("✅ Data loading complete!")
    
    # ========================================
    # Model Configuration
    # ========================================
    print_header("Model Configurations")
    
    models_config = {
        'MLP_Small': MLP(hidden_sizes=[32, 24, 16], input_channels=1, input_size=28),
        'MLP_Large': MLP(hidden_sizes=[128, 32], input_channels=1, input_size=28),
        'CNN_Small': SimpleCNN(num_filters=16, input_channels=1, input_size=28),
        'CNN_Large': SimpleCNN(num_filters=32, input_channels=1, input_size=28),
        # 'ResNet18': ResNet18CIFAR(input_channels=1)
    }
    
    for name, model in models_config.items():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {num_params:,} parameters")
    
    # ========================================
    # Training Configuration
    # ========================================
    training_config = {
        'batch_size': 128,
        # 'learning_rate': 0.001,
        'learning_rate': 0.0005,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        # 'max_epochs': 50,
        'max_epochs': 50,
        'val_split': 0.1,
        'random_seed': 42,
        'data_cap_rate': 20,  # Using 1/20 of data for overfitting experiments
        'lr_scheduler': 'ReduceLROnPlateau',
        'lr_scheduler_patience': 5,
        'lr_scheduler_factor': 0.5,
        
        # ========================================
        # Early Stopping Configuration
        # ========================================
        # Set 'early_stopping_enabled' to False to disable early stopping
        # and train for the full 'max_epochs'
        'early_stopping_enabled': True,          # Toggle: True to enable, False to disable
        'early_stopping_patience': 10,           # Wait N epochs before stopping
        'early_stopping_mode': 'min',            # 'min' for loss, 'max' for accuracy
        'early_stopping_restore_best': True      # Restore best weights when stopping
    }
    
    print("\nTraining Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    # Highlight early stopping status
    if training_config['early_stopping_enabled']:
        print(f"\n⚡ Early stopping: ENABLED (patience={training_config['early_stopping_patience']})")
    else:
        print(f"\n⏱️  Early stopping: DISABLED (will train for full {training_config['max_epochs']} epochs)")
    
    # ========================================
    # Setup Results Tracking
    # ========================================
    visualizer = ResultsVisualizer(output_dir='results')
    
    results = {}
    test_accuracies = {}
    
    # ========================================
    # Train Each Model
    # ========================================
    for model_name, model in models_config.items():
        print_header(f"Training {model_name}")
        
        # Setup loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        
        # Setup learning rate scheduler (ReduceLROnPlateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',           # minimize validation loss
            factor=0.5,           # multiply LR by 0.5
            patience=5,           # wait 5 epochs before reducing
            min_lr=1e-6          # minimum learning rate
        )
        
        # Setup early stopping
        early_stopping = None
        if training_config['early_stopping_enabled']:
            early_stopping = EarlyStopping(
                patience=training_config['early_stopping_patience'],
                mode=training_config['early_stopping_mode'],
                restore_best_weights=training_config['early_stopping_restore_best'],
                verbose=True
            )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_epochs=training_config['max_epochs'],
            scheduler=scheduler
        )
        
        # Train the model with early stopping
        history = trainer.train(early_stopping=early_stopping)
        
        # Test the model
        test_acc = trainer.test()
        
        # Store results
        results[model_name] = history
        test_accuracies[model_name] = test_acc
        
        # Save individual training curves
        print(f"\n📊 Generating plots for {model_name}...")
        visualizer.plot_training_curves(history, model_name)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(visualizer.get_save_dir(), f'{model_name}_final.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'test_accuracy': test_acc,
            'config': training_config
        }, checkpoint_path)
        print(f"💾 Saved model checkpoint: {model_name}_final.pth")
    
    # ========================================
    # Generate Comparison Plots
    # ========================================
    print_header("Generating Comparison Plots")
    visualizer.plot_comparison(results)
    
    # ========================================
    # Save Summary
    # ========================================
    print_header("Saving Summary")
    
    # Build model configs dynamically
    model_configs_summary = {}
    for name, model in models_config.items():
        num_params = sum(p.numel() for p in model.parameters())
        model_configs_summary[name] = f"{num_params:,} parameters"
    
    # Collect early stopping info and training times for summary
    early_stopping_info = {}
    training_times = {}
    for model_name, history in results.items():
        early_stopping_info[model_name] = {
            'early_stopped': history.get('early_stopped', False),
            'stopped_epoch': history.get('stopped_epoch', None),
            'best_epoch': history.get('best_epoch', None)
        }
        total_time = history.get('total_training_time', 0.0)
        training_times[model_name] = {
            'seconds': round(total_time, 2),
            'minutes': round(total_time / 60, 2),
            'hours': round(total_time / 3600, 3)
        }
    
    summary = {
        'test_accuracies': test_accuracies,
        'training_times': training_times,
        'training_config': training_config,
        'model_configs': model_configs_summary,
        'dataset': data_module.name,
        'dataset_sizes': {
            'train': len(data_module.train_dataset),
            'validation': len(data_module.val_dataset),
            'test': len(data_module.test_dataset)
        },
        'early_stopping_info': early_stopping_info,
        'phase': '1.1 - Early Stopping',
        'notes': f'Overfitting experiment with data_cap_rate={data_module.data_cap_rate}. Early stopping {"enabled" if training_config["early_stopping_enabled"] else "disabled"}'
    }
    
    visualizer.save_summary(summary)
    
    # Print final summary
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print("\nTest Accuracies:")
    for model_name, acc in test_accuracies.items():
        print(f"  {model_name}: {acc:.2f}%")
    
    print("\nTraining Times:")
    for model_name, times in training_times.items():
        print(f"  {model_name}: {times['minutes']:.2f} minutes ({times['seconds']:.2f}s)")
    
    print(f"\n📁 All results saved to: {visualizer.get_save_dir()}")
    print("\nGenerated files:")
    for model_name in models_config.keys():
        print(f"  - {model_name}_training_curves.png")
    print("  - model_comparison.png")
    print("  - summary.json")
    print("  - Model checkpoints (.pth files)")
    
    print("\n✨ Next steps:")
    print("  - Phase 1.2: Implement Double Descent experiments")
    print("  - Phase 2: Create Jupyter Notebooks for analysis")


if __name__ == '__main__':
    main()

