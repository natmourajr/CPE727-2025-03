"""
Multi-Layer Perceptron (MLP) for image classification

A simple feedforward neural network that flattens the input image
and processes it through fully connected layers.
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for image classification
    
    Args:
        hidden_sizes (list): List of hidden layer sizes
        dropout (float): Dropout probability for regularization
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        input_size (int): Input image dimension (28 for FashionMNIST, 32 for CIFAR-10)
    """
    
    def __init__(self, hidden_sizes=[512, 256], dropout=0.2, num_classes=10, input_channels=3, input_size=32):
        super(MLP, self).__init__()
        
        layers = []
        flat_input_size = input_size * input_size * input_channels  # Flattened image size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(flat_input_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            flat_input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(flat_input_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.flatten(x)
        return self.network(x)


def get_mlp_variants():
    """Get different MLP configurations for experimentation"""
    return {
        'MLP_Small': MLP(hidden_sizes=[128, 64]),
        'MLP_Medium': MLP(hidden_sizes=[512, 256]),
        'MLP_Large': MLP(hidden_sizes=[1024, 512, 256]),
    }

