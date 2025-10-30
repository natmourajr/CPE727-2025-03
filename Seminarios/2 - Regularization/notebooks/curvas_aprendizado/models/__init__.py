"""Model architectures"""
from .mlp import MLP
from .cnn import SimpleCNN
from .resnet import ResNet18CIFAR

__all__ = ['MLP', 'SimpleCNN', 'ResNet18CIFAR']

