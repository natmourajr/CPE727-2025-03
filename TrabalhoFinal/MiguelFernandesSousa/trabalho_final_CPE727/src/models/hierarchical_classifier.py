"""
Hierarchical Classifier for Fashion MNIST

Two-stage hierarchical architecture to handle class confusion patterns
identified in EDA analysis.

Stage 1: Coarse grouping (10 classes → 3 groups)
    - Group "Tops": T-shirt (0), Pullover (2), Coat (4), Shirt (6)
    - Group "Footwear": Sandal (5), Sneaker (7), Ankle boot (9)
    - Group "Other": Trouser (1), Dress (3), Bag (8)

Stage 2: Three specialist classifiers
    - Stage 2a: 4-way classification within "Tops" (hardest group)
    - Stage 2b: 3-way classification within "Footwear"
    - Stage 2c: 3-way classification within "Other" (easiest group)

Motivation (from EDA):
    - T-shirt vs Shirt confusion is high (similar spatial patterns)
    - Footwear items are well-separated from clothing
    - Hierarchical approach allows specialized models per difficulty level
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, Any, Optional
import pickle
from pathlib import Path


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """
    Two-stage hierarchical classifier for Fashion MNIST

    Attributes:
        stage1_model: Classifier for coarse grouping (3-way)
        stage2a_model: Specialist for "Tops" group (4-way)
        stage2b_model: Specialist for "Footwear" group (3-way)
        stage2c_model: Specialist for "Other" group (3-way)
    """

    # Group definitions (from EDA analysis)
    GROUP_DEFINITIONS = {
        "Tops": [0, 2, 4, 6],  # T-shirt, Pullover, Coat, Shirt
        "Footwear": [5, 7, 9],  # Sandal, Sneaker, Ankle boot
        "Other": [1, 3, 8],  # Trouser, Dress, Bag
    }

    # Reverse mapping: class → group name
    CLASS_TO_GROUP = {}
    for group_name, classes in GROUP_DEFINITIONS.items():
        for cls in classes:
            CLASS_TO_GROUP[cls] = group_name

    # Group name to integer mapping for stage 1
    GROUP_TO_INT = {"Tops": 0, "Footwear": 1, "Other": 2}
    INT_TO_GROUP = {v: k for k, v in GROUP_TO_INT.items()}

    def __init__(
        self,
        stage1_model,
        stage2a_model,  # Tops
        stage2b_model,  # Footwear
        stage2c_model,  # Other
    ):
        """
        Initialize hierarchical classifier

        Args:
            stage1_model: Model for stage 1 (3-way group classification)
            stage2a_model: Model for Tops group (4-way)
            stage2b_model: Model for Footwear group (3-way)
            stage2c_model: Model for Other group (3-way)
        """
        self.stage1_model = stage1_model
        self.stage2a_model = stage2a_model
        self.stage2b_model = stage2b_model
        self.stage2c_model = stage2c_model

        # Will be set during fit
        self.classes_ = None
        self.n_features_in_ = None

    def _convert_labels_to_groups(self, y: np.ndarray) -> np.ndarray:
        """
        Convert original 10-class labels to 3-group labels

        Args:
            y: Original labels (0-9)

        Returns:
            Group labels (0=Tops, 1=Footwear, 2=Other)
        """
        y_groups = np.array([self.GROUP_TO_INT[self.CLASS_TO_GROUP[label]] for label in y])
        return y_groups

    def _get_samples_for_group(
        self, X: np.ndarray, y: np.ndarray, group_name: str
    ) -> tuple:
        """
        Filter samples belonging to a specific group

        Args:
            X: Features
            y: Original labels (0-9)
            group_name: Name of group ('Tops', 'Footwear', 'Other')

        Returns:
            X_group, y_group (with original labels)
        """
        group_classes = self.GROUP_DEFINITIONS[group_name]
        mask = np.isin(y, group_classes)
        return X[mask], y[mask]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the hierarchical classifier

        1. Train stage 1 on all data with group labels
        2. Split data by predicted groups
        3. Train stage 2 specialists on their respective groups

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,) with values 0-9

        Returns:
            self
        """
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        # Stage 1: Train coarse grouping
        print(f"\n[Stage 1] Training coarse grouping (10 → 3 classes)...")
        y_groups = self._convert_labels_to_groups(y)
        self.stage1_model.fit(X, y_groups)

        group_counts = np.bincount(y_groups)
        print(f"  Group distribution:")
        for group_idx, group_name in self.INT_TO_GROUP.items():
            print(f"    {group_name}: {group_counts[group_idx]} samples")

        # Stage 2a: Train Tops specialist
        print(f"\n[Stage 2a] Training Tops specialist (4-way)...")
        X_tops, y_tops = self._get_samples_for_group(X, y, "Tops")
        self.stage2a_model.fit(X_tops, y_tops)
        print(f"  Trained on {len(X_tops)} samples")
        print(f"  Classes: {np.unique(y_tops)}")

        # Stage 2b: Train Footwear specialist
        print(f"\n[Stage 2b] Training Footwear specialist (3-way)...")
        X_footwear, y_footwear = self._get_samples_for_group(X, y, "Footwear")
        self.stage2b_model.fit(X_footwear, y_footwear)
        print(f"  Trained on {len(X_footwear)} samples")
        print(f"  Classes: {np.unique(y_footwear)}")

        # Stage 2c: Train Other specialist
        print(f"\n[Stage 2c] Training Other specialist (3-way)...")
        X_other, y_other = self._get_samples_for_group(X, y, "Other")
        self.stage2c_model.fit(X_other, y_other)
        print(f"  Trained on {len(X_other)} samples")
        print(f"  Classes: {np.unique(y_other)}")

        print(f"\n✓ Hierarchical classifier training completed")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes using two-stage hierarchy

        1. Stage 1: Predict group (Tops, Footwear, Other)
        2. Stage 2: Route to appropriate specialist for final class

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,) with values 0-9
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        # Stage 1: Predict groups
        group_predictions = self.stage1_model.predict(X)

        # Stage 2: Route to specialists
        for group_idx, group_name in self.INT_TO_GROUP.items():
            # Find samples predicted for this group
            mask = group_predictions == group_idx

            if not np.any(mask):
                continue

            X_group = X[mask]

            # Select appropriate specialist
            if group_name == "Tops":
                class_predictions = self.stage2a_model.predict(X_group)
            elif group_name == "Footwear":
                class_predictions = self.stage2b_model.predict(X_group)
            else:  # Other
                class_predictions = self.stage2c_model.predict(X_group)

            predictions[mask] = class_predictions

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using two-stage hierarchy

        Combines:
        - P(group | X) from stage 1
        - P(class | X, group) from stage 2

        Final: P(class | X) = P(group | X) × P(class | X, group)

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilities (n_samples, 10)
        """
        n_samples = X.shape[0]
        n_classes = 10
        probas = np.zeros((n_samples, n_classes))

        # Stage 1: Get group probabilities
        group_probas = self.stage1_model.predict_proba(X)  # (n_samples, 3)

        # Stage 2: For each group, get class probabilities
        for group_idx, group_name in self.INT_TO_GROUP.items():
            # Select appropriate specialist
            if group_name == "Tops":
                specialist = self.stage2a_model
            elif group_name == "Footwear":
                specialist = self.stage2b_model
            else:  # Other
                specialist = self.stage2c_model

            # Get class probabilities within this group
            class_probas_in_group = specialist.predict_proba(X)  # (n_samples, n_classes_in_group)

            # Map to full 10-class space
            group_classes = self.GROUP_DEFINITIONS[group_name]
            for local_idx, global_class in enumerate(group_classes):
                # P(class | X) = P(group | X) × P(class | X, group)
                probas[:, global_class] = (
                    group_probas[:, group_idx] * class_probas_in_group[:, local_idx]
                )

        return probas

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy

        Args:
            X: Features
            y: True labels

        Returns:
            Accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_stage_metrics(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get detailed metrics for each stage

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary with stage-wise metrics
        """
        from sklearn.metrics import accuracy_score

        # Stage 1 accuracy
        y_groups_true = self._convert_labels_to_groups(y)
        y_groups_pred = self.stage1_model.predict(X)
        stage1_acc = accuracy_score(y_groups_true, y_groups_pred)

        # Stage 2 accuracies (only on samples correctly routed by stage 1)
        metrics = {
            "stage1_accuracy": stage1_acc,
            "stage2_accuracies": {},
            "overall_accuracy": self.score(X, y),
        }

        for group_idx, group_name in self.INT_TO_GROUP.items():
            # Get samples that truly belong to this group
            X_group, y_group = self._get_samples_for_group(X, y, group_name)

            if len(X_group) == 0:
                continue

            # Select specialist
            if group_name == "Tops":
                specialist = self.stage2a_model
            elif group_name == "Footwear":
                specialist = self.stage2b_model
            else:  # Other
                specialist = self.stage2c_model

            # Accuracy within group
            y_pred_group = specialist.predict(X_group)
            acc = accuracy_score(y_group, y_pred_group)
            metrics["stage2_accuracies"][group_name] = acc

        return metrics

    def save(self, filepath: Path):
        """Save hierarchical classifier to disk"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        print(f"  ✓ Hierarchical classifier saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> "HierarchicalClassifier":
        """Load hierarchical classifier from disk"""
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            classifier = pickle.load(f)

        print(f"  ✓ Hierarchical classifier loaded from {filepath}")
        return classifier


if __name__ == "__main__":
    # Test hierarchical classifier
    print("Testing Hierarchical Classifier...\n")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from src.models import LogisticRegressionSoftmax, RandomForest

    # Create synthetic 10-class dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=10,
        n_clusters_per_class=1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create hierarchical classifier
    print("Creating hierarchical classifier...")
    print("  Stage 1: Logistic Softmax")
    print("  Stage 2a (Tops): Random Forest")
    print("  Stage 2b (Footwear): Logistic Softmax")
    print("  Stage 2c (Other): Logistic Softmax")

    hierarchical = HierarchicalClassifier(
        stage1_model=LogisticRegressionSoftmax(max_iter=1000, verbose=0),
        stage2a_model=RandomForest(n_estimators=50, max_depth=10, n_jobs=-1),
        stage2b_model=LogisticRegressionSoftmax(max_iter=1000, verbose=0),
        stage2c_model=LogisticRegressionSoftmax(max_iter=1000, verbose=0),
    )

    # Train
    print("\nTraining hierarchical classifier...")
    hierarchical.fit(X_train, y_train)

    # Predict
    print("\nEvaluating...")
    y_pred = hierarchical.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✓ Overall accuracy: {acc:.4f}")

    # Get stage metrics
    stage_metrics = hierarchical.get_stage_metrics(X_test, y_test)
    print(f"\nStage metrics:")
    print(f"  Stage 1 (grouping) accuracy: {stage_metrics['stage1_accuracy']:.4f}")
    print(f"  Stage 2 specialist accuracies:")
    for group_name, acc in stage_metrics['stage2_accuracies'].items():
        print(f"    {group_name}: {acc:.4f}")

    # Test probabilities
    probas = hierarchical.predict_proba(X_test[:5])
    print(f"\n✓ Probabilities (first 5 samples):")
    print(probas)
    print(f"  Shape: {probas.shape}")
    print(f"  Sum per row (should be ~1.0): {probas.sum(axis=1)}")

    print("\n✓ Hierarchical Classifier implemented successfully!")
