"""
Training pipeline using PyTorch Lightning.

Implements 5x2 cross-validation from the paper.
Includes MLflow integration for experiment tracking and model versioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import mlflow
import logging

from src.evaluation import compute_metrics, MetricsTracker
from src.data.dataset import collate_fn_windows, collate_fn_full

logger = logging.getLogger(__name__)


class IARAClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for IARA classification.
    
    Wraps any model and provides training/validation logic.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 4,
        class_names: List[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize classifier.
        
        Args:
            model: PyTorch model to train
            num_classes: Number of classes
            class_names: List of class names
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function with optional class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_preds = []
        self.val_targets = []
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'class_weights'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y, _ = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """Validation step."""
        x, y, _ = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Predictions
        preds = torch.argmax(logits, dim=1)
        
        # Store for epoch-level metrics
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss}
    
    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch."""
        if not self.val_preds:
            return
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.val_preds).numpy()
        all_targets = torch.cat(self.val_targets).numpy()
        
        # Compute metrics
        metrics = compute_metrics(
            all_targets,
            all_preds,
            self.num_classes,
            self.class_names
        )
        
        # Log metrics
        self.log('val_sp', metrics['sp'], prog_bar=True)
        self.log('val_balanced_acc', metrics['balanced_accuracy'], prog_bar=True)
        self.log('val_f1', metrics['f1_macro'], prog_bar=True)
        
        # Clear for next epoch
        self.val_preds = []
        self.val_targets = []
    
    def test_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """Test step."""
        x, y, _ = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Predictions
        preds = torch.argmax(logits, dim=1)
        
        return {
            'test_loss': loss,
            'preds': preds.cpu(),
            'targets': y.cpu()
        }
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler (OneCycleLR from paper)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate * 10,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


class CrossValidationTrainer:
    """
    Implements 5x2 cross-validation from the paper (Section V-D).

    Includes MLflow integration for comprehensive experiment tracking:
    - Automatic hyperparameter logging
    - Model versioning and registry
    - Artifact storage (models, metrics, plots)
    - Experiment comparison UI
    """

    def __init__(
        self,
        dataset,
        model_fn: callable,
        num_classes: int = 4,
        class_names: List[str] = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        early_stopping_patience: int = 10,
        output_dir: Path = Path("experiments"),
        use_class_weights: bool = True,
        experiment_name: str = "iara_classification",
        use_mlflow: bool = True,
        mlflow_tracking_uri: str = None,
    ):
        """
        Initialize cross-validation trainer.

        Args:
            dataset: IARA dataset
            model_fn: Function that returns a new model instance
            num_classes: Number of classes
            class_names: List of class names
            batch_size: Batch size
            max_epochs: Maximum epochs per fold
            learning_rate: Learning rate
            weight_decay: Weight decay
            early_stopping_patience: Patience for early stopping
            output_dir: Output directory for results
            use_class_weights: Whether to use class weights
            experiment_name: Name for MLflow experiment
            use_mlflow: Whether to use MLflow logging (default: True)
            mlflow_tracking_uri: MLflow tracking URI (default: file:./mlruns)
        """
        self.dataset = dataset
        self.model_fn = model_fn
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.output_dir = Path(output_dir)
        self.use_class_weights = use_class_weights
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup MLflow
        if self.use_mlflow:
            self.mlflow_tracking_uri = mlflow_tracking_uri or f"file://{self.output_dir}/mlruns"
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow tracking URI: {self.mlflow_tracking_uri}")
            logger.info(f"MLflow experiment: {self.experiment_name}")

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(num_classes, class_names)
    
    def create_data_loaders(
        self,
        train_indices: List[int],
        val_indices: List[int],
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for train and validation.

        Args:
            train_indices: Indices for training
            val_indices: Indices for validation

        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        # Determine which collate function to use based on dataset
        if self.dataset.use_windows:
            collate_fn = collate_fn_windows
        else:
            collate_fn = collate_fn_full

        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader
    
    def run_fold(
        self,
        fold_idx: int,
        train_indices: List[int],
        val_indices: List[int],
    ) -> Dict:
        """
        Run one fold of cross-validation.
        
        Args:
            fold_idx: Fold index
            train_indices: Training indices
            val_indices: Validation indices
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'=' * 70}")
        print(f"Fold {fold_idx + 1}")
        print(f"{'=' * 70}")
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_indices, val_indices)
        
        # Get class weights if needed
        class_weights = None
        if self.use_class_weights:
            class_weights = self.dataset.get_class_weights()
            print(f"Class weights: {class_weights.numpy()}")
        
        # Create model
        model = self.model_fn()
        
        # Create Lightning module
        pl_model = IARAClassifier(
            model=model,
            num_classes=self.num_classes,
            class_names=self.class_names,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            class_weights=class_weights,
        )
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / f"fold_{fold_idx}",
            filename='best-{epoch:02d}-{val_sp:.4f}',
            monitor='val_sp',
            mode='max',
            save_top_k=1,
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_sp',
            patience=self.early_stopping_patience,
            mode='max',
            verbose=True,
        )
        
        # Loggers
        tb_logger = TensorBoardLogger(
            save_dir=self.output_dir,
            name=f"fold_{fold_idx}",
        )

        loggers = [tb_logger]

        # Add MLflow logger if enabled
        if self.use_mlflow:
            mlflow_logger = MLFlowLogger(
                experiment_name=self.experiment_name,
                tracking_uri=self.mlflow_tracking_uri,
                run_name=f"fold_{fold_idx}",
            )
            loggers.append(mlflow_logger)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=loggers,
            enable_progress_bar=True,
            deterministic=True,
        )
        
        # Train
        trainer.fit(pl_model, train_loader, val_loader)
        
        # Test on validation set
        results = trainer.test(pl_model, val_loader, ckpt_path='best')
        
        # Get predictions
        pl_model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y, _ = batch
                x = x.to(pl_model.device)
                logits = pl_model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Compute metrics
        metrics = compute_metrics(
            all_targets,
            all_preds,
            self.num_classes,
            self.class_names
        )
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  SP: {metrics['sp']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_macro']:.4f}")

        # Log metrics to MLflow
        if self.use_mlflow:
            mlflow.log_metrics({
                'fold_sp': metrics['sp'],
                'fold_balanced_accuracy': metrics['balanced_accuracy'],
                'fold_f1_macro': metrics['f1_macro'],
                'fold_accuracy': metrics['accuracy'],
            })

            # Log confusion matrix as artifact
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(all_targets, all_preds)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - Fold {fold_idx + 1}')

            # Save and log to MLflow
            cm_path = self.output_dir / f"fold_{fold_idx}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            # Log best model checkpoint
            if checkpoint_callback.best_model_path:
                mlflow.log_artifact(checkpoint_callback.best_model_path)

        # Track metrics
        self.metrics_tracker.add_fold(all_targets, all_preds)

        return {
            'metrics': metrics,
            'preds': all_preds,
            'targets': all_targets,
            'checkpoint': str(checkpoint_callback.best_model_path),
        }
    
    def run_5x2_cv(self) -> Dict:
        """
        Run 5x2 cross-validation.
        
        Returns:
            Dictionary with aggregated results
        """
        print("=" * 70)
        print("Running 5x2 Cross-Validation")
        print("=" * 70)
        
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        
        all_results = []
        
        # 5 iterations
        for iteration in range(5):
            print(f"\n{'#' * 70}")
            print(f"Iteration {iteration + 1}/5")
            print(f"{'#' * 70}")
            
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Split into two folds
            split_point = n_samples // 2
            fold1_indices = indices[:split_point].tolist()
            fold2_indices = indices[split_point:].tolist()
            
            # Train on fold1, test on fold2
            results_1 = self.run_fold(
                fold_idx=iteration * 2,
                train_indices=fold1_indices,
                val_indices=fold2_indices,
            )
            all_results.append(results_1)
            
            # Train on fold2, test on fold1
            results_2 = self.run_fold(
                fold_idx=iteration * 2 + 1,
                train_indices=fold2_indices,
                val_indices=fold1_indices,
            )
            all_results.append(results_2)
        
        # Aggregate results
        print("\n" + "=" * 70)
        print("5x2 Cross-Validation Results")
        print("=" * 70)
        print(self.metrics_tracker.format_summary())
        
        # Save results
        import json
        results_path = self.output_dir / "cv_results.json"
        with open(results_path, 'w') as f:
            summary = self.metrics_tracker.get_summary()
            # Convert to serializable format
            summary_serializable = {
                k: [float(v[0]), float(v[1])] for k, v in summary.items()
            }
            json.dump(summary_serializable, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_path}")
        
        return {
            'summary': self.metrics_tracker.get_summary(),
            'all_results': all_results,
        }
