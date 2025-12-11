"""
Evaluation metrics following the IARA paper.

Implements (Section V-D):
- Sum-Product Index (SP) - geometric mean metric
- Balanced Accuracy (macro-averaged)
- F1-Score (macro-averaged)
- Confusion Matrix
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def sum_product_index(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Compute Sum-Product Index (SP) from the paper.
    
    From equation (1) in Section V-D:
    SP = sqrt((1/k * sum(E_i)) * (product(E_i))^(1/k))
    
    where E_i is the detection probability (recall) for class i
    and k is the number of classes.
    
    This metric penalizes classification errors more strongly than accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Sum-Product Index value
    """
    # Compute per-class recall (detection probability)
    recalls = []
    
    for class_idx in range(num_classes):
        # True positives for this class
        mask_true = (y_true == class_idx)
        mask_pred = (y_pred == class_idx)
        
        tp = np.sum(mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        # Recall = TP / (TP + FN)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        
        recalls.append(recall)
    
    recalls = np.array(recalls)
    
    # Compute SP following equation (1)
    # Arithmetic mean
    arithmetic_mean = np.mean(recalls)
    
    # Geometric mean (handle zeros)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    geometric_mean = np.exp(np.mean(np.log(recalls + epsilon)))
    
    # SP is the geometric mean of arithmetic and geometric means
    sp = np.sqrt(arithmetic_mean * geometric_mean)
    
    return sp


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics from the paper.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        class_names: Optional list of class names
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "sp": sum_product_index(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred)
    
    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        
        # Precision, Recall, F1 for each class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_label = class_names[class_idx] if class_names else f"class_{class_idx}"
        metrics[f"precision_{class_label}"] = precision
        metrics[f"recall_{class_label}"] = recall
        metrics[f"f1_{class_label}"] = f1
    
    return metrics


def format_metrics(metrics: Dict[str, float], decimal_places: int = 4) -> str:
    """
    Format metrics dict as a readable string.
    
    Args:
        metrics: Dictionary of metrics
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    
    # Main metrics first
    main_metrics = ["sp", "balanced_accuracy", "accuracy", "f1_macro", "f1_weighted"]
    
    for key in main_metrics:
        if key in metrics:
            value = metrics[key] * 100  # Convert to percentage
            lines.append(f"{key:20s}: {value:6.{decimal_places}f}%")
    
    # Per-class metrics
    per_class_keys = [k for k in metrics.keys() if k.startswith(("precision_", "recall_", "f1_"))]
    
    if per_class_keys:
        lines.append("\nPer-class metrics:")
        for key in sorted(per_class_keys):
            value = metrics[key] * 100
            lines.append(f"  {key:30s}: {value:6.{decimal_places}f}%")
    
    return "\n".join(lines)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: If True, normalize by true label counts
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        cmap = 'Blues'
    else:
        fmt = 'd'
        cmap = 'Blues'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    return report


class MetricsTracker:
    """
    Track metrics across multiple folds/iterations.
    
    Implements the 5x2 cross-validation tracking from the paper.
    """
    
    def __init__(self, num_classes: int, class_names: List[str]):
        """
        Initialize metrics tracker.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names
        
        self.fold_metrics: List[Dict[str, float]] = []
        self.confusion_matrices: List[np.ndarray] = []
    
    def add_fold(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Add results from one fold.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        metrics = compute_metrics(y_true, y_pred, self.num_classes, self.class_names)
        self.fold_metrics.append(metrics)
        
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices.append(cm)
    
    def get_summary(self) -> Dict[str, Tuple[float, float]]:
        """
        Get summary statistics (mean ± std) across folds.
        
        Returns:
            Dictionary with (mean, std) tuples for each metric
        """
        if not self.fold_metrics:
            return {}
        
        summary = {}
        
        # Get all metric keys
        metric_keys = self.fold_metrics[0].keys()
        
        for key in metric_keys:
            values = [fold[key] for fold in self.fold_metrics]
            summary[key] = (np.mean(values), np.std(values))
        
        return summary
    
    def format_summary(self, decimal_places: int = 2) -> str:
        """
        Format summary as string (mean ± std).
        
        Args:
            decimal_places: Number of decimal places
            
        Returns:
            Formatted string
        """
        summary = self.get_summary()
        lines = []
        
        # Main metrics
        main_metrics = ["sp", "balanced_accuracy", "accuracy", "f1_macro"]
        
        for key in main_metrics:
            if key in summary:
                mean, std = summary[key]
                mean_pct = mean * 100
                std_pct = std * 100
                lines.append(f"{key:20s}: {mean_pct:6.{decimal_places}f} ± {std_pct:4.{decimal_places}f}%")
        
        return "\n".join(lines)
    
    def get_average_confusion_matrix(self, normalize: bool = True) -> np.ndarray:
        """
        Get average confusion matrix across folds.
        
        Args:
            normalize: If True, normalize by true label counts
            
        Returns:
            Average confusion matrix
        """
        if not self.confusion_matrices:
            return np.zeros((self.num_classes, self.num_classes))
        
        avg_cm = np.mean(self.confusion_matrices, axis=0)
        
        if normalize:
            avg_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)
        
        return avg_cm
    
    def reset(self):
        """Reset all tracked metrics."""
        self.fold_metrics = []
        self.confusion_matrices = []
