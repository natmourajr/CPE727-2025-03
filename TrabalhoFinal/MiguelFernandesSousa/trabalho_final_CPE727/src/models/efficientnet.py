"""
EfficientNet models using timm.
"""

import torch.nn as nn
from typing import Optional

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def get_efficientnet_b0(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Get EfficientNet-B0 model adapted for IARA.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to use ImageNet pre-trained weights
        
    Returns:
        EfficientNet-B0 model
    """
    if not HAS_TIMM:
        raise ImportError("timm is required. Install with: pip install timm")
    
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    
    return model


def get_efficientnet_b1(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Get EfficientNet-B1 model adapted for IARA.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to use ImageNet pre-trained weights
        
    Returns:
        EfficientNet-B1 model
    """
    if not HAS_TIMM:
        raise ImportError("timm is required. Install with: pip install timm")
    
    model = timm.create_model(
        'efficientnet_b1',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    
    return model
