# Early Stopping Guide

## Overview

This implementation follows the early stopping algorithm from Goodfellow et al.'s Deep Learning textbook (see `instructions.md` lines 14-40).

## How to Toggle Early Stopping

### Enable Early Stopping (Default)

In `main.py`, set:
```python
'early_stopping_enabled': True,
'early_stopping_patience': 10,
'early_stopping_mode': 'min',
'early_stopping_restore_best': True
```

### Disable Early Stopping

To train for the full number of epochs without early stopping:
```python
'early_stopping_enabled': False,
```

When disabled, the model will train for the complete `max_epochs` specified in the configuration.

## Configuration Options

| Parameter | Description | Values |
|-----------|-------------|--------|
| `early_stopping_enabled` | Toggle early stopping on/off | `True` or `False` |
| `early_stopping_patience` | Number of epochs to wait for improvement | Any positive integer (default: 10) |
| `early_stopping_mode` | Metric to monitor | `'min'` (for loss) or `'max'` (for accuracy) |
| `early_stopping_restore_best` | Restore best model weights when stopping | `True` or `False` |

## How It Works

The algorithm:
1. **Monitors** validation loss after each epoch
2. **Saves** the best model weights when improvement is found
3. **Counts** consecutive epochs without improvement
4. **Stops** training when counter reaches patience
5. **Restores** best model weights (if `restore_best_weights=True`)

### Example

```
Epoch 1: val_loss=0.50 → Save as best (counter=0)
Epoch 2: val_loss=0.45 → Improved! Save as best (counter=0)
Epoch 3: val_loss=0.46 → No improvement (counter=1)
Epoch 4: val_loss=0.47 → No improvement (counter=2)
...
Epoch 12: val_loss=0.48 → No improvement (counter=10)
→ Early stopping triggered! Restore weights from epoch 2
```

## Output Summary

The training summary includes:

### 1. Training Times
```json
"training_times": {
  "MLP_Small": {
    "seconds": 45.23,
    "minutes": 0.75,
    "hours": 0.013
  }
}
```

### 2. Early Stopping Info
```json
"early_stopping_info": {
  "MLP_Small": {
    "early_stopped": true,
    "stopped_epoch": 25,
    "best_epoch": 15
  }
}
```

### 3. Visual Markers on Plots

Training curve plots show:
- **Green dashed line**: Best model epoch
- **Orange dotted line**: Early stopping epoch (if triggered)

## Use Cases

### For Experiments

**Enable early stopping** when:
- You want to prevent overfitting
- Training time is limited
- You're doing hyperparameter search

**Disable early stopping** when:
- Studying double descent phenomena
- Need consistent epoch counts across experiments
- Analyzing full training dynamics

## Algorithm Reference

This implementation matches the textbook algorithm:

| Textbook Variable | Our Implementation |
|-------------------|-------------------|
| `p` (patience) | `patience` parameter |
| `v` (best validation error) | `best_score` (negated) |
| `j` (counter) | `counter` |
| `θ*` (best parameters) | `best_weights` |
| `i*` (best step) | `best_epoch` |

See `instructions.md` for the full algorithm pseudocode.

