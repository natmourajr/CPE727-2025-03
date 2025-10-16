"""
Early Stopping Implementation

Implements the early stopping algorithm from Goodfellow et al. (Deep Learning Book)
following the pseudo-code in instructions.md lines 14-40.

Algorithm Overview:
- Let p be the patience (number of times to observe worsening validation error)
- Let v be the best validation error seen so far
- Let θ* be the best parameters
- Let j be the counter of consecutive non-improvements

The algorithm stops training when j >= p and returns the best parameters θ*.
"""

import copy
import torch


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    
    This implementation follows the textbook algorithm:
    - Tracks best validation metric (v in algorithm)
    - Counts consecutive epochs without improvement (j in algorithm)
    - Saves best model weights (θ* in algorithm)
    - Stops training when counter reaches patience (j >= p)
    - Restores best weights when stopping
    
    Args:
        patience (int): Number of epochs to wait after last improvement before stopping.
                       Corresponds to 'p' in the textbook algorithm.
        mode (str): One of 'min' or 'max'. In 'min' mode, training stops when the
                   metric stops decreasing; in 'max' mode, stops when increasing.
        restore_best_weights (bool): Whether to restore model weights from the epoch
                                    with the best value of the monitored metric.
                                    Returns θ* from the algorithm.
        verbose (bool): Whether to print messages when improvement occurs or stopping triggers.
    
    Example:
        >>> early_stopping = EarlyStopping(patience=10, mode='min')
        >>> for epoch in range(max_epochs):
        >>>     train_loss = train_epoch()
        >>>     val_loss = validate()
        >>>     early_stopping(val_loss, model)
        >>>     if early_stopping.early_stop:
        >>>         print(f"Early stopping triggered at epoch {epoch}")
        >>>         break
    """
    
    def __init__(self, patience=7, mode='min', restore_best_weights=True, verbose=True):
        self.patience = patience  # p in algorithm
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Algorithm state variables
        self.counter = 0  # j in algorithm (consecutive non-improvements)
        self.best_score = None  # v in algorithm (best validation error, negated for max mode)
        self.early_stop = False  # Flag to signal training should stop
        self.best_weights = None  # θ* in algorithm (best model parameters)
        self.best_epoch = 0  # i* in algorithm (epoch when best model was found)
        self.current_epoch = 0  # i in algorithm (current training step)
        
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, metric, model, epoch=None):
        """
        Check if training should stop based on the validation metric.
        
        Implements the core logic from the textbook algorithm:
        IF v' < v THEN
            j ← 0
            θ* ← θ
            v ← v'
        ELSE
            j ← j + 1
        
        Args:
            metric (float): Current validation metric (e.g., validation loss)
            model (torch.nn.Module): The model being trained
            epoch (int, optional): Current epoch number for tracking
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        # Convert metric to score (higher is better)
        # For 'min' mode: negate so lower metric = higher score
        # For 'max' mode: use as-is so higher metric = higher score
        score = -metric if self.mode == 'min' else metric
        
        # First call: initialize best score
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = self.current_epoch
            self.save_checkpoint(model)
            if self.verbose:
                print(f"Early stopping: Initial best score: {metric:.6f} at epoch {self.current_epoch}")
            return
        
        # Check if current score is better than best score
        # This implements: IF v' < v (in the case of minimizing v)
        if score > self.best_score:
            # Improvement found!
            # Algorithm lines 29-33:
            # j ← 0
            # θ* ← θ
            # i* ← i
            # v ← v'
            self.best_score = score
            self.best_epoch = self.current_epoch
            self.save_checkpoint(model)
            self.counter = 0  # Reset counter (j ← 0)
            
            if self.verbose:
                print(f"Early stopping: Improvement found! New best score: {metric:.6f} at epoch {self.current_epoch}")
        else:
            # No improvement
            # Algorithm lines 35:
            # j ← j + 1
            self.counter += 1
            
            if self.verbose:
                print(f"Early stopping: No improvement. Counter: {self.counter}/{self.patience}")
            
            # Check if we should stop
            # Algorithm line 25: WHILE j < p (stop when j >= p)
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    # Algorithm line 38: Return best parameters θ*
                    self.restore_checkpoint(model)
                    if self.verbose:
                        print(f"Early stopping: Restoring best weights from epoch {self.best_epoch}")
    
    def save_checkpoint(self, model):
        """
        Save model weights (θ* ← θ)
        
        Args:
            model (torch.nn.Module): Model to save
        """
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())
    
    def restore_checkpoint(self, model):
        """
        Restore best model weights (Return θ*)
        
        Args:
            model (torch.nn.Module): Model to restore weights to
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
    
    def state_dict(self):
        """
        Return state dictionary for saving/loading
        
        Returns:
            dict: Dictionary containing early stopping state
        """
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'current_epoch': self.current_epoch,
            'early_stop': self.early_stop
        }
    
    def load_state_dict(self, state_dict):
        """
        Load state from dictionary
        
        Args:
            state_dict (dict): Dictionary containing early stopping state
        """
        self.counter = state_dict.get('counter', 0)
        self.best_score = state_dict.get('best_score', None)
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.early_stop = state_dict.get('early_stop', False)

