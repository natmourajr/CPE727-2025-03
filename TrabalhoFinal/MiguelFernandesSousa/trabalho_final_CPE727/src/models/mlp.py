"""
MLP models for IARA experiments.

Following paper's MLP configuration (Section VI-A):
- 2 hidden layers
- Batch normalization
- Dropout
- Various activation functions
"""

import torch
import torch.nn as nn
from typing import List, Optional


class IARAMLP(nn.Module):
    """
    MLP architecture based on IARA paper baseline.
    
    Architecture (from Table 7, MEL/LOFAR configuration):
    - Hidden layers: [32, 16] neurons
    - ReLU activation
    - BatchNorm1D
    - Dropout (0.2 for MEL, 0.0 for LOFAR)
    - Output: Sigmoid activation
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 4,
        hidden_sizes: List[int] = [32, 16],
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize IARA MLP.
        
        Args:
            input_size: Size of flattened input features
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'prelu', 'leaky_relu')
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 to disable)
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif activation == "prelu":
                layers.append(nn.PReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(prev_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, features) or (batch, channels, height, width)
            
        Returns:
            Output logits (batch, num_classes)
        """
        # Flatten if needed
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Output
        x = self.output(x)
        
        return x


def get_iara_mlp(
    input_size: int,
    num_classes: int = 4,
    config: str = "mel",
    pretrained: bool = False,
) -> IARAMLP:
    """
    Get IARA MLP with pre-defined configurations from paper.
    
    Args:
        input_size: Size of flattened input features
        num_classes: Number of classes
        config: Configuration ('mel' or 'lofar')
        pretrained: Not used (kept for API consistency)
        
    Returns:
        IARAMLP model
    """
    if config == "mel":
        # MEL configuration from Table 7
        return IARAMLP(
            input_size=input_size,
            num_classes=num_classes,
            hidden_sizes=[32, 16],
            activation="relu",
            use_batch_norm=True,
            dropout_rate=0.2,
        )
    elif config == "lofar":
        # LOFAR configuration from Table 7
        return IARAMLP(
            input_size=input_size,
            num_classes=num_classes,
            hidden_sizes=[32, 16],
            activation="relu",
            use_batch_norm=True,
            dropout_rate=0.0,  # No dropout for LOFAR
        )
    else:
        raise ValueError(f"Unknown config: {config}. Choose 'mel' or 'lofar'.")
