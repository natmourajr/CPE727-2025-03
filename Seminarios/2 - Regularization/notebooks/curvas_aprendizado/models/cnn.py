"""
Convolutional Neural Network (CNN) for image classification

A simple CNN with convolutional blocks and fully connected layers.
Based on PyTorch CIFAR-10 tutorial.

Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for image classification
    
    Architecture:
        - 3 Convolutional blocks (conv -> relu -> maxpool)
        - 2 Fully connected layers
        - Batch normalization and dropout for regularization
    
    Args:
        num_filters (int): Base number of filters (doubled in each block)
        dropout (float): Dropout probability
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        input_size (int): Input image dimension (28 for FashionMNIST, 32 for CIFAR-10)
    """
    
    def __init__(self, num_filters=64, dropout=0.25, num_classes=10, input_channels=3, input_size=32):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters * 4)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Calculate FC layer input size after 3 pooling layers
        # Each pooling reduces size by 2: size -> size/2 -> size/4 -> size/8
        fc_input_size = num_filters * 4 * (input_size // 8) * (input_size // 8)
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv block 1: size -> size/2
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2: size/2 -> size/4
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3: size/4 -> size/8
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_cnn_variants():
    """Get different CNN configurations for experimentation"""
    return {
        'CNN_Small': SimpleCNN(num_filters=32),
        'CNN_Medium': SimpleCNN(num_filters=64),
        'CNN_Large': SimpleCNN(num_filters=128),
    }

