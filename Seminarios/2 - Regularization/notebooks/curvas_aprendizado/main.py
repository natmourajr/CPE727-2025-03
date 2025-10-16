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
from training.pipeline import TrainingPipeline, setup_training_config
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
    """Main training loop using the modular TrainingPipeline"""
    
    print_header("Model Training - Phase 1.1")
    print("Training MLP and CNN models with Early Stopping")
    print("Following textbook algorithm from Goodfellow et al.")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # ========================================
    # Training Configuration
    # ========================================
    training_config = setup_training_config(
        batch_size=128,
        learning_rate=0.0005,
        optimizer='Adam',
        loss_function='CrossEntropyLoss',
        max_epochs=50,
        val_split=0.1,
        random_seed=42,
        data_cap_rate=20,  # Using 1/20 of data for overfitting experiments
        lr_scheduler='ReduceLROnPlateau',
        lr_scheduler_patience=5,
        lr_scheduler_factor=0.5,
        
        # ========================================
        # Early Stopping Configuration
        # ========================================
        # Set 'early_stopping_enabled' to False to disable early stopping
        # and train for the full 'max_epochs'
        early_stopping_enabled=True,          # Toggle: True to enable, False to disable
        early_stopping_patience=10,           # Wait N epochs before stopping
        early_stopping_mode='min',            # 'min' for loss, 'max' for accuracy
        early_stopping_restore_best=True      # Restore best weights when stopping
    )
    
    # ========================================
    # Create and Run Training Pipeline
    # ========================================
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        dataset_name='FashionMNIST',
        config=training_config,
        verbose=True  # Keep verbose output for main script
    )
    
    # Setup the pipeline (load data, create models, etc.)
    pipeline.setup()
    
    # Train all models
    results = pipeline.train()
    
    # Save results and generate plots
    pipeline.save_results()
    
    # Print final summary
    pipeline.print_summary()
    
    print("\n✨ Next steps:")
    print("  - Phase 1.2: Implement Double Descent experiments")
    print("  - Phase 2: Create Jupyter Notebooks for analysis")


if __name__ == '__main__':
    main()

