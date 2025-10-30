"""
Training Pipeline Module

Modular training pipeline that can be used both in main.py and Jupyter notebooks.
Extracts and organizes the training logic for reusability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
from datetime import datetime

# Import our modules
from data.fashion_mnist_loader import FashionMNISTDataModule
from data.cifar10_loader import CIFAR10DataModule
from data.svhn_loader import SVHNDataModule
from models.mlp import MLP
from models.cnn import SimpleCNN
from models.resnet import ResNet18CIFAR
from training.trainer import Trainer
from training.early_stopping import EarlyStopping
from visualization.plots import ResultsVisualizer


def setup_device():
    """
    Setup device for training (CUDA > MPS > CPU)
    
    Returns:
        torch.device: Device to use for training
    """
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
    
    return device


def load_dataset(dataset_name, data_dir='./data', batch_size=64, val_split=0.1, 
                data_cap_rate=None, num_workers=2):
    """
    Load dataset based on name
    
    Args:
        dataset_name (str): Name of dataset ('FashionMNIST', 'CIFAR10', 'SVHN')
        data_dir (str): Directory containing data
        batch_size (int): Batch size for data loaders
        val_split (float): Validation split ratio
        data_cap_rate (int): Rate to cap data (1/N of full dataset)
        num_workers (int): Number of worker processes
        
    Returns:
        DataModule: Dataset module with prepared data
    """
    dataset_classes = {
        'FashionMNIST': FashionMNISTDataModule,
        'CIFAR10': CIFAR10DataModule,
        'SVHN': SVHNDataModule
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_classes.keys())}")
    
    # Create data module
    data_module = dataset_classes[dataset_name](
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers
    )
    
    # Add data cap if specified
    if data_cap_rate is not None:
        data_module.data_cap_rate = data_cap_rate
    
    # Prepare and setup data
    print(f"Loading {dataset_name} dataset...")
    data_module.prepare_data()
    data_module.setup()
    
    print("✅ Data loading complete!")
    return data_module


def create_models(input_channels, input_size, dataset_name='FashionMNIST'):
    """
    Create model configurations
    
    Args:
        input_channels (int): Number of input channels
        input_size (int): Input image size
        dataset_name (str): Name of dataset (affects model selection)
        
    Returns:
        dict: Dictionary mapping model names to model instances
    """
    models_config = {
        'MLP_Small': MLP(hidden_sizes=[32, 24, 16], input_channels=input_channels, input_size=input_size),
        'MLP_Large': MLP(hidden_sizes=[128, 32], input_channels=input_channels, input_size=input_size),
        'CNN_Small': SimpleCNN(num_filters=16, input_channels=input_channels, input_size=input_size),
        'CNN_Large': SimpleCNN(num_filters=32, input_channels=input_channels, input_size=input_size),
    }
    
    # Add ResNet for CIFAR10/SVHN (32x32 images)
    if input_size == 32 and dataset_name in ['CIFAR10', 'SVHN']:
        models_config['ResNet18'] = ResNet18CIFAR(input_channels=input_channels)
    
    return models_config


def setup_training_config(**kwargs):
    """
    Setup training configuration with defaults
    
    Args:
        **kwargs: Override default configuration values
        
    Returns:
        dict: Training configuration
    """
    default_config = {
        'batch_size': 128,
        'learning_rate': 0.0005,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'max_epochs': 50,
        'val_split': 0.1,
        'random_seed': 42,
        'data_cap_rate': 20,
        'lr_scheduler': 'ReduceLROnPlateau',
        'lr_scheduler_patience': 5,
        'lr_scheduler_factor': 0.5,
        'early_stopping_enabled': True,
        'early_stopping_patience': 10,
        'early_stopping_mode': 'min',
        'early_stopping_restore_best': True
    }
    
    # Update with provided kwargs
    default_config.update(kwargs)
    return default_config


def train_single_model(model, model_name, train_loader, val_loader, test_loader, 
                      device, config, early_stopping_config=None, verbose=True):
    """
    Train a single model
    
    Args:
        model: PyTorch model
        model_name (str): Name of the model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        config (dict): Training configuration
        early_stopping_config (dict): Early stopping configuration
        verbose (bool): Whether to print detailed training progress
        
    Returns:
        tuple: (history, test_accuracy)
    """
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'],
        min_lr=1e-6
    )
    
    # Setup early stopping
    early_stopping = None
    if early_stopping_config and early_stopping_config.get('enabled', False):
        early_stopping = EarlyStopping(
            patience=early_stopping_config['patience'],
            mode=early_stopping_config['mode'],
            restore_best_weights=early_stopping_config['restore_best'],
            verbose=verbose
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
        max_epochs=config['max_epochs'],
        scheduler=scheduler,
        verbose=verbose
    )
    
    # Train the model
    history = trainer.train(early_stopping=early_stopping)
    
    # Test the model
    test_acc = trainer.test()
    
    return history, test_acc


def train_all_models(models_config, data_module, device, training_config, verbose=True):
    """
    Train all models in the configuration
    
    Args:
        models_config (dict): Dictionary of model name -> model instance
        data_module: Dataset module with data loaders
        device: Device to train on
        training_config (dict): Training configuration
        verbose (bool): Whether to print detailed training progress
        
    Returns:
        dict: Results dictionary with histories and test accuracies
    """
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    results = {}
    test_accuracies = {}
    
    # Print model information
    print(f"\n{'='*60}")
    print("  Model Configurations")
    print(f"{'='*60}")
    for name, model in models_config.items():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {num_params:,} parameters")
    
    # Train each model
    for model_name, model in models_config.items():
        # Setup early stopping config
        early_stopping_config = None
        if training_config.get('early_stopping_enabled', False):
            early_stopping_config = {
                'enabled': True,
                'patience': training_config['early_stopping_patience'],
                'mode': training_config['early_stopping_mode'],
                'restore_best': training_config['early_stopping_restore_best']
            }
        
        # Train model
        history, test_acc = train_single_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            config=training_config,
            early_stopping_config=early_stopping_config,
            verbose=verbose
        )
        
        # Store results
        results[model_name] = history
        test_accuracies[model_name] = test_acc
    
    return results, test_accuracies


class TrainingPipeline:
    """
    Training pipeline that encapsulates the entire training process
    
    This class provides a clean interface for training models with different
    configurations, making it easy to use in both scripts and notebooks.
    """
    
    def __init__(self, dataset_name='FashionMNIST', config=None, verbose=True):
        """
        Initialize training pipeline
        
        Args:
            dataset_name (str): Name of dataset to use
            config (dict): Training configuration (uses defaults if None)
            verbose (bool): Whether to print detailed training progress
        """
        self.dataset_name = dataset_name
        self.config = config or setup_training_config()
        self.verbose = verbose
        self.data_module = None
        self.device = None
        self.models_config = None
        self.results = None
        self.test_accuracies = None
        self.visualizer = None
        
    def setup(self, data_dir='./data', batch_size=None, val_split=None, 
              data_cap_rate=None, num_workers=2):
        """
        Setup the training pipeline (load data, create models, etc.)
        
        Args:
            data_dir (str): Directory containing data
            batch_size (int): Batch size (uses config default if None)
            val_split (float): Validation split (uses config default if None)
            data_cap_rate (int): Data cap rate (uses config default if None)
            num_workers (int): Number of worker processes
        """
        # Use config defaults if not specified
        batch_size = batch_size or self.config['batch_size']
        val_split = val_split or self.config['val_split']
        data_cap_rate = data_cap_rate or self.config.get('data_cap_rate')
        
        # Setup device
        self.device = setup_device()
        
        # Load dataset
        self.data_module = load_dataset(
            dataset_name=self.dataset_name,
            data_dir=data_dir,
            batch_size=batch_size,
            val_split=val_split,
            data_cap_rate=data_cap_rate,
            num_workers=num_workers
        )
        
        # Determine input parameters based on dataset
        input_channels, input_size = self._get_input_params()
        
        # Create models
        self.models_config = create_models(
            input_channels=input_channels,
            input_size=input_size,
            dataset_name=self.dataset_name
        )
        
        # Setup visualizer
        self.visualizer = ResultsVisualizer(output_dir='results')
        
        print(f"\n✅ Pipeline setup complete!")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Device: {self.device}")
        print(f"   Models: {list(self.models_config.keys())}")
    
    def _get_input_params(self):
        """Get input channels and size based on dataset"""
        if self.dataset_name == 'FashionMNIST':
            return 1, 28  # Grayscale 28x28
        elif self.dataset_name in ['CIFAR10', 'SVHN']:
            return 3, 32  # RGB 32x32
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def train(self, early_stopping_enabled=None):
        """
        Train all models
        
        Args:
            early_stopping_enabled (bool): Override early stopping setting
            
        Returns:
            dict: Results dictionary
        """
        if self.data_module is None:
            raise RuntimeError("Pipeline not setup. Call setup() first.")
        
        # Override early stopping if specified
        if early_stopping_enabled is not None:
            self.config['early_stopping_enabled'] = early_stopping_enabled
        
        # Print configuration
        self._print_config()
        
        # Train all models
        self.results, self.test_accuracies = train_all_models(
            models_config=self.models_config,
            data_module=self.data_module,
            device=self.device,
            training_config=self.config,
            verbose=self.verbose
        )
        
        return self.results
    
    def _print_config(self):
        """Print training configuration"""
        print(f"\n{'='*60}")
        print("  Training Configuration")
        print(f"{'='*60}")
        
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        # Highlight early stopping status
        if self.config['early_stopping_enabled']:
            print(f"\n⚡ Early stopping: ENABLED (patience={self.config['early_stopping_patience']})")
        else:
            print(f"\n⏱️  Early stopping: DISABLED (will train for full {self.config['max_epochs']} epochs)")
    
    def get_results(self):
        """Get training results"""
        if self.results is None:
            raise RuntimeError("No results available. Run train() first.")
        return self.results, self.test_accuracies
    
    def save_results(self, output_dir=None):
        """
        Save training results and generate plots
        
        Args:
            output_dir (str): Output directory (uses visualizer default if None)
        """
        if self.results is None:
            raise RuntimeError("No results available. Run train() first.")
        
        # Generate individual training curves
        for model_name, history in self.results.items():
            print(f"\n📊 Generating plots for {model_name}...")
            self.visualizer.plot_training_curves(history, model_name)
        
        # Generate comparison plots
        print(f"\n📊 Generating comparison plots...")
        self.visualizer.plot_comparison(self.results)
        
        # Save model checkpoints
        for model_name, model in self.models_config.items():
            checkpoint_path = os.path.join(self.visualizer.get_save_dir(), f'{model_name}_final.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'history': self.results[model_name],
                'test_accuracy': self.test_accuracies[model_name],
                'config': self.config
            }, checkpoint_path)
            print(f"💾 Saved model checkpoint: {model_name}_final.pth")
        
        # Save summary
        self._save_summary()
        
        print(f"\n📁 All results saved to: {self.visualizer.get_save_dir()}")
    
    def _save_summary(self):
        """Save experiment summary"""
        # Collect early stopping info and training times
        early_stopping_info = {}
        training_times = {}
        
        for model_name, history in self.results.items():
            early_stopping_info[model_name] = {
                'early_stopped': history.get('early_stopped', False),
                'stopped_epoch': history.get('stopped_epoch', None),
                'best_epoch': history.get('best_epoch', None)
            }
        total_time = history.get('total_training_time', 0.0)
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        training_times[model_name] = f"{minutes}m {seconds}s"
        
        # Build model configs summary
        model_configs_summary = {}
        for name, model in self.models_config.items():
            num_params = sum(p.numel() for p in model.parameters())
            model_configs_summary[name] = f"{num_params:,} parameters"
        
        summary = {
            'test_accuracies': self.test_accuracies,
            'training_times': training_times,
            'training_config': self.config,
            'model_configs': model_configs_summary,
            'dataset': self.data_module.name,
            'dataset_sizes': {
                'train': len(self.data_module.train_dataset),
                'validation': len(self.data_module.val_dataset),
                'test': len(self.data_module.test_dataset)
            },
            'early_stopping_info': early_stopping_info,
            'phase': '1.1 - Early Stopping',
            'notes': f'Overfitting experiment with data_cap_rate={self.data_module.data_cap_rate}. Early stopping {"enabled" if self.config["early_stopping_enabled"] else "disabled"}'
        }
        
        self.visualizer.save_summary(summary)
    
    def print_summary(self):
        """Print training summary"""
        if self.results is None:
            raise RuntimeError("No results available. Run train() first.")
        
        print("\n" + "="*60)
        print("  TRAINING COMPLETE!")
        print("="*60)
        
        print("\nTest Accuracies:")
        for model_name, acc in self.test_accuracies.items():
            print(f"  {model_name}: {acc:.2f}%")
        
        print("\nTraining Times:")
        for model_name, history in self.results.items():
            total_time = history.get('total_training_time', 0.0)
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print(f"  {model_name}: {minutes}m {seconds}s")
        
        print("\nEarly Stopping Info:")
        for model_name, history in self.results.items():
            if history.get('early_stopped', False):
                print(f"  {model_name}: Stopped at epoch {history.get('stopped_epoch')}, best at epoch {history.get('best_epoch')}")
            else:
                print(f"  {model_name}: Completed all epochs")
