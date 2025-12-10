"""
ResNet models using torchvision and timm.

Adapted for single-channel spectrograms.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not installed. Install with: pip install timm")


def get_resnet18(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Get ResNet18 model adapted for IARA.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for grayscale spectrogram)
        pretrained: Whether to use ImageNet pre-trained weights
        
    Returns:
        ResNet18 model
    """
    if not HAS_TIMM:
        raise ImportError("timm is required. Install with: pip install timm")
    
    model = timm.create_model(
        'resnet18',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    
    return model


def get_resnet34(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Get ResNet34 model adapted for IARA.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to use ImageNet pre-trained weights
        
    Returns:
        ResNet34 model
    """
    if not HAS_TIMM:
        raise ImportError("timm is required. Install with: pip install timm")
    
    model = timm.create_model(
        'resnet34',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    
    return model


def get_resnet50(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Get ResNet50 model adapted for IARA.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to use ImageNet pre-trained weights
        
    Returns:
        ResNet50 model
    """
    if not HAS_TIMM:
        raise ImportError("timm is required. Install with: pip install timm")
    
    model = timm.create_model(
        'resnet50',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    
    return model
