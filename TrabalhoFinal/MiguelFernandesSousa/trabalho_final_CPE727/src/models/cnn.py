"""
CNN models for IARA experiments.

Following paper's CNN configuration (Section VI-A):
- 2 convolutional layers
- Batch normalization
- Max pooling
- Dropout
- Fully connected layers
"""

import torch
import torch.nn as nn
from typing import Optional


class IARACNN(nn.Module):
    """
    CNN architecture based on IARA paper baseline.
    
    Architecture (from Table 7, MEL configuration):
    - Conv1: 1024 filters, kernel=5, padding=1, LeakyReLU
    - Conv2: 128 filters, kernel=5, padding=1, LeakyReLU
    - BatchNorm2D after each conv
    - MaxPool2D: 2x2
    - Dropout: 0.4
    - FC: 128 units, ReLU
    - Output: num_classes with Sigmoid
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 4,
        conv1_filters: int = 1024,
        conv2_filters: int = 128,
        fc_units: int = 128,
        kernel_size: int = 5,
        padding: int = 1,
        pool_size: int = 2,
        dropout_rate: float = 0.4,
        activation: str = "leaky_relu",
    ):
        """
        Initialize IARA CNN.
        
        Args:
            input_channels: Number of input channels (1 for single spectrogram)
            num_classes: Number of output classes
            conv1_filters: Filters in first conv layer
            conv2_filters: Filters in second conv layer
            fc_units: Units in fully connected layer
            kernel_size: Kernel size for conv layers
            padding: Padding for conv layers
            pool_size: Pooling window size
            dropout_rate: Dropout rate
            activation: Activation function ('relu' or 'leaky_relu')
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Choose activation
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()
        
        # Feature extractor
        self.conv1 = nn.Conv2d(
            input_channels,
            conv1_filters,
            kernel_size=kernel_size,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        
        self.conv2 = nn.Conv2d(
            conv1_filters,
            conv2_filters,
            kernel_size=kernel_size,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout_conv = nn.Dropout2d(p=dropout_rate)
        
        # Calculate size after convolutions
        # This will be computed dynamically in forward pass
        self.fc1 = None
        self.fc_units = fc_units
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        
        # Output layer
        self.output = nn.Linear(fc_units, num_classes)
    
    def _initialize_fc(self, x: torch.Tensor):
        """Initialize FC layer based on conv output size."""
        if self.fc1 is None:
            # Get flattened size
            batch_size = x.size(0)
            flattened_size = x.view(batch_size, -1).size(1)
            
            self.fc1 = nn.Linear(flattened_size, self.fc_units).to(x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Output logits (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Flatten
        batch_size = x.size(0)
        
        # Initialize FC if needed
        if self.fc1 is None:
            self._initialize_fc(x)
        
        x = x.view(batch_size, -1)
        
        # Fully connected
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout_fc(x)
        
        # Output
        x = self.output(x)
        
        return x


def get_iara_cnn(
    num_classes: int = 4,
    config: str = "mel",
    pretrained: bool = False,
) -> IARACNN:
    """
    Get IARA CNN with pre-defined configurations from paper.
    
    Args:
        num_classes: Number of classes
        config: Configuration ('mel' or 'lofar')
        pretrained: Not used (kept for API consistency)
        
    Returns:
        IARACNN model
    """
    if config == "mel":
        # MEL configuration from Table 7
        return IARACNN(
            num_classes=num_classes,
            conv1_filters=1024,
            conv2_filters=128,
            fc_units=128,
            kernel_size=5,
            padding=1,
            pool_size=2,
            dropout_rate=0.4,
            activation="leaky_relu",
        )
    elif config == "lofar":
        # LOFAR configuration from Table 7
        return IARACNN(
            num_classes=num_classes,
            conv1_filters=512,
            conv2_filters=128,
            fc_units=128,
            kernel_size=5,
            padding=1,
            pool_size=(4, 2),  # Different pooling for LOFAR
            dropout_rate=0.4,
            activation="relu",
        )
    else:
        raise ValueError(f"Unknown config: {config}. Choose 'mel' or 'lofar'.")
