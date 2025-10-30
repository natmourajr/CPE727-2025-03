"""
Basic Training Loop for PyTorch Models

Implements a standard training loop with validation and testing.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time


class Trainer:
    """
    Basic trainer for PyTorch models
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu or cuda)
        max_epochs: Maximum number of epochs to train
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        max_epochs=100,
        scheduler=None,
        verbose=True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.verbose = verbose
        
        # Track training metrics
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': [],
            'learning_rates': [],
            'early_stopped': False,
            'stopped_epoch': None,
            'best_epoch': None,
            'total_training_time': 0.0
        }
    
    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validate the model
        
        Returns:
            tuple: (val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss / (pbar.n + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, early_stopping=None):
        """
        Full training loop with optional early stopping
        
        Args:
            early_stopping: EarlyStopping instance or None. If provided, training
                          will stop when validation metric stops improving.
        
        Returns:
            dict: Training history
        """
        print(f"Training for up to {self.max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if early_stopping:
            print(f"Early stopping enabled: patience={early_stopping.patience}, mode={early_stopping.mode}")
        
        best_val_acc = 0.0
        training_start_time = time.time()  # Track total training time
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            
            if self.verbose:
                print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Track time
            epoch_time = time.time() - epoch_start
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(current_lr)
            
            # Update learning rate scheduler if present
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                # Get updated learning rate after scheduler step
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr and self.verbose:
                    print(f"⚡ Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.history['best_epoch'] = epoch + 1
            
            # Check early stopping
            if early_stopping:
                early_stopping(val_loss, self.model, epoch=epoch + 1)
                if early_stopping.early_stop:
                    if self.verbose:
                        print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                        print(f"   No improvement for {early_stopping.patience} consecutive epochs")
                        print(f"   Best epoch was: {early_stopping.best_epoch}")
                        if early_stopping.restore_best_weights:
                            print(f"   ✅ Restored best model weights from epoch {early_stopping.best_epoch}")
                    
                    # Update history with early stopping info
                    self.history['early_stopped'] = True
                    self.history['stopped_epoch'] = epoch + 1
                    self.history['best_epoch'] = early_stopping.best_epoch
                    break
            
            # Print summary (only if verbose)
            if self.verbose:
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        
        # Calculate total training time
        total_time = time.time() - training_start_time
        self.history['total_training_time'] = total_time
        
        # Final summary
        if early_stopping and early_stopping.early_stop:
            if self.verbose:
                print(f"\nTraining stopped early at epoch {self.history['stopped_epoch']}")
                print(f"Best epoch: {self.history['best_epoch']}")
        else:
            if self.verbose:
                print(f"\nCompleted all {self.max_epochs} epochs")
        
        if self.verbose:
            print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
            print(f"Total Training Time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
        
        return self.history
    
    def test(self):
        """
        Test the model on test set
        
        Returns:
            float: Test accuracy
        """
        print("\nEvaluating on test set...")
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        test_acc = 100. * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        return test_acc

