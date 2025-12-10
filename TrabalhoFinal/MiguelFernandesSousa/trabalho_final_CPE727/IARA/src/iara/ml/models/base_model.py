"""
This module provides base classes for building and serializing PyTorch models.

Classes:
    - SerializableModel: A base class for serializable PyTorch models, providing methods for saving
        and loading models using pickle.
    - BaseModel: A base class for PyTorch models with serialization support, inheriting from both
        torch.nn.Module and SerializableModel.
"""
import abc

import pickle
import torch

class Serializable:
    """
    A base class for serializable objects.

    Provides methods for saving and loading the objects using pickle.
    """
    def save(self, file_path: str) -> None:
        """
        Save the model to a file using pickle.

        Parameters:
            - file_path (str): The path to the file to save the model.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str) -> 'Serializable':
        """
        Load a model from a file using pickle.

        Parameters:
            - file_path (str): The path to the file containing the saved model.

        Returns:
            SerializableModel: The loaded model.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

class BaseModel(torch.nn.Module, Serializable):
    """ A base class for PyTorch models with serialization support."""

    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

    def __str__(self) -> str:
        """
        Return a string representation of the model.

        Returns:
            str: A string containing the name of the model class.
        """
        return self.name

    @abc.abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """ Keeping abstract method of torch.nn.Module

        Args:
            data (torch.Tensor): Teste input data

        Returns:
            torch.Tensor: prediction
        """
