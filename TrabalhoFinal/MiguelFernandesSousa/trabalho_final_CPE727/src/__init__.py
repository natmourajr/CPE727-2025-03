"""
IARA Deep Learning Package

PyTorch implementation for underwater acoustic target recognition
using the IARA dataset.
"""

from .features import (
    LOFARExtractor,
    MELExtractor,
    SpectrogramNormalizer,
    resample_audio
)

from .dataset import (
    IARADataset,
    create_iara_dataloaders
)

__version__ = '0.1.0'
__all__ = [
    'LOFARExtractor',
    'MELExtractor',
    'SpectrogramNormalizer',
    'resample_audio',
    'IARADataset',
    'create_iara_dataloaders'
]
