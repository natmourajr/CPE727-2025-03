"""
ConvNeXt models using timm.
"""

import torch.nn as nn
from typing import Optional

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def get_convnext_tiny(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Get ConvNeXt-Tiny model adapted for IARA.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to use ImageNet pre-trained weights
        
    Returns:
        ConvNeXt-Tiny model
    """
    if not HAS_TIMM:
        raise ImportError("timm is required. Install with: pip install timm")
    
    model = timm.create_model(
        'convnext_tiny',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    
    return model


def get_convnext_small(
    num_classes: int = 4,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Get ConvNeXt-Small model adapted for IARA.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to use ImageNet pre-trained weights
        
    Returns:
        ConvNeXt-Small model
    """
    if not HAS_TIMM:
        raise ImportError("timm is required. Install with: pip install timm")
    
    model = timm.create_model(
        'convnext_small',
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    
    return model
