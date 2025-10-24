import sys
import os
# --- REQUIRED PATH FIX ---
# This forces the directory containing main.py onto the Python search path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -------------------------

import torch
import torch.optim as optim
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import time

# --- MUST USE SIMPLE, DOTLESS IMPORTS NOW ---
# The manual path fix allows these simple imports to work
from data_loader import get_cifar10_data
from model import get_cnn_model
from train_utils import train_one_epoch, evaluate_model

# --- HYPERPARAMETERS ---
LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM_FACTOR = 0.9
NUM_EPOCHS = 50
BATCH_SIZE = 128

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# --- Plotting Function ---
def plot_comparison(history_sgd, history_momentum, metric_name, title):
    """Plots a metric (loss or accuracy) for both experiments."""
    
    epochs = range(1, len(history_sgd[metric_name]) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot SGD (No Momentum)
    plt.plot(epochs, history_sgd[metric_name], label='SGD (No Momentum)', color='red', linestyle='--')
    
    # Plot SGD with Momentum
    plt.plot(epochs, history_momentum[metric_name], label='SGD (With Momentum)', color='blue', linestyle='-')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'results/{metric_name}_comparison.png')
    plt.show()

# --- Experiment Runner Function ---
def run_experiment(name, model, optimizer, scheduler, train_loader, val_loader, criterion, device):
    print(f"\n--- Starting Experiment: {name} ---")
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_wts = None
    
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        # Step the scheduler (adjust LR)
        scheduler.step()

        # Log Metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Experiment {name} finished. Best Val Accuracy: {best_val_acc:.4f}. Total Time: {elapsed_time:.2f} seconds.")
    return history, best_model_wts, elapsed_time

# --- Main Execution Block ---
if __name__ == '__main__':
    # ----------------------------------------------------
    # Setup: Device Configuration for Apple Silicon (M-series)
    # ----------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS backend for GPU acceleration. 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA backend.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    
    # Remaining Setup
    train_loader, val_loader, test_loader = get_cifar10_data(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    
    results_table = {}

    # ----------------------------------------------------
    # Experiment 1: SGD without Momentum
    # ----------------------------------------------------
    model_sgd = get_cnn_model().to(device)
    optimizer_sgd = optim.SGD(
        model_sgd.parameters(), 
        lr=LR, 
        momentum=0.0,
        weight_decay=WEIGHT_DECAY
    )
    scheduler_sgd = optim.lr_scheduler.StepLR(optimizer_sgd, step_size=20, gamma=0.1)

    history_sgd, weights_sgd, time_sgd = run_experiment(
        "SGD_NoMomentum", model_sgd, optimizer_sgd, scheduler_sgd, 
        train_loader, val_loader, criterion, device
    )

    # ----------------------------------------------------
    # Experiment 2: SGD with Momentum
    # ----------------------------------------------------
    model_momentum = get_cnn_model().to(device)
    optimizer_momentum = optim.SGD(
        model_momentum.parameters(), 
        lr=LR, 
        momentum=MOMENTUM_FACTOR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler_momentum = optim.lr_scheduler.StepLR(optimizer_momentum, step_size=20, gamma=0.1)
    
    history_momentum, weights_momentum, time_momentum = run_experiment(
        "SGD_WithMomentum", model_momentum, optimizer_momentum, scheduler_momentum, 
        train_loader, val_loader, criterion, device
    )
    
    # ----------------------------------------------------
    # Final Evaluation and Comparison
    # ----------------------------------------------------
    
    # Evaluate best models on Test Set
    model_sgd.load_state_dict(weights_sgd)
    test_loss_sgd, test_acc_sgd = evaluate_model(model_sgd, test_loader, criterion, device)
    
    model_momentum.load_state_dict(weights_momentum)
    test_loss_mom, test_acc_mom = evaluate_model(model_momentum, test_loader, criterion, device)

    # Print Final Results Table
    print("\n" + "="*50)
    print("--- Final Comparison Results ---")
    print(f"{'Optimizer':<20}{'Test Accuracy':<15}{'Total Time (s)':<15}")
    print("-" * 50)
    print(f"{'SGD (No Momentum)':<20}{test_acc_sgd:.4f}{'':<4}{time_sgd:.2f}")
    print(f"{'SGD (With Momentum)':<20}{test_acc_mom:.4f}{'':<4}{time_momentum:.2f}")
    print("="*50)

    # Plot the Curves
    print("\nGenerating comparison plots...")
    plot_comparison(history_sgd, history_momentum, 'val_loss', 'Validation Loss over Epochs: SGD vs. SGD with Momentum')
    plot_comparison(history_sgd, history_momentum, 'val_acc', 'Validation Accuracy over Epochs: SGD vs. SGD with Momentum')
    
    print("Plots saved to the 'results' folder.")