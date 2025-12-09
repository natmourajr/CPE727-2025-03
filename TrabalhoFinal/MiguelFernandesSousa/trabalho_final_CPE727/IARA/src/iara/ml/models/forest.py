"""
Module containing a Forest based models.
"""
import sklearn.ensemble as sk_ensemble

import torch

import iara.ml.models.base_model as iara_model

class RandomForestModel(iara_model.BaseModel):
    """A class representing a Random Forest model."""

    def __init__(self, n_estimators=100, max_depth=None):
        """
        Initialize the Random Forest model.

        Args:
            n_estimators (int): The number of trees in the forest.
            max_depth (int): The maximum depth of the trees. If None, the trees are expanded until
                all leaves are pure.
        """
        super().__init__()
        self.random_forest = sk_ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                                max_depth=max_depth,
                                                                random_state=42,
                                                                n_jobs=40,
                                                                class_weight='balanced')

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Predicted class labels.
        """
        data_flat = data.view(data.size(0), -1)
        predictions = self.random_forest.predict(data_flat.cpu().numpy())
        return torch.tensor(predictions)

    def fit(self, samples: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Fit the Random Forest model to the training data.

        Args:
            samples (torch.Tensor): Data sample
            targets (torch.Tensor): targets
        """
        samples_flat = samples.view(samples.size(0), -1)
        self.random_forest.fit(samples_flat, targets)
