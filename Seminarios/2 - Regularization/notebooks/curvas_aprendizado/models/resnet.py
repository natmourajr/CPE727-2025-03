"""
ResNet-18 adapted for small images (CIFAR-10, FashionMNIST, etc.)

Modified ResNet-18 architecture for small images using PyTorch's standard ResNet18.
Reference: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
"""

import torch.nn as nn
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for small images (28x28 or 32x32)
    
    Uses PyTorch's standard ResNet-18 with modifications:
    1. Smaller initial conv layer (kernel_size=3 instead of 7, stride=1 instead of 2)
    2. Remove max pooling layer (too aggressive for small images)
    3. Configurable input channels (1 for grayscale, 3 for RGB)
    4. Modified final layer for specified number of classes
    
    Args:
        num_classes (int): Number of output classes (default: 10)
        input_channels (int): Number of input channels - 1 for grayscale, 3 for RGB (default: 3)
        pretrained (bool): Whether to use pretrained weights (default: False)
    
    Reference: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    """
    
    def __init__(self, num_classes=10, input_channels=3, pretrained=False):
        super(ResNet18CIFAR, self).__init__()
        
        # Load ResNet-18 architecture without pretrained weights
        if pretrained:
            print("Warning: Using pretrained weights may not be optimal for small images")
        
        self.model = resnet18(weights=None)
        
        # Modify first convolutional layer for small images and configurable input channels
        # Standard ResNet: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Small image adapted: Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Remove the max pooling layer (too aggressive for 28x28 or 32x32 images)
        self.model.maxpool = nn.Identity()
        
        # Modify final fully connected layer for specified number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.model(x)


def get_resnet_variants():
    """
    Get different ResNet-18 configurations for experimentation
    
    Returns:
        dict: Dictionary of model name to model instance
    """
    return {
        'ResNet18_RGB': ResNet18CIFAR(input_channels=3),
        'ResNet18_Grayscale': ResNet18CIFAR(input_channels=1),
    }

