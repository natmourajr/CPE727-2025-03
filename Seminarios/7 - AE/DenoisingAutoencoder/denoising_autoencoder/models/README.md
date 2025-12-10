# Denoising Autoencoder (DAE)

PyTorch implementation of Denoising Autoencoders for image reconstruction.

## Models

### DenoisingAutoencoder
Base class for denoising autoencoders with encode/decode interface.

### ConvolutionalDenoisingAutoencoder
Convolutional implementation with:
- 3-block encoder with Conv2d → BatchNorm → ReLU → MaxPool
- Configurable latent space dimension
- 3-block decoder with ConvTranspose2d → BatchNorm → ReLU
- Final Conv2d output layer with Sigmoid activation

## Installation

```bash
pip install -e .
```

## Usage

```python
from DAE.denoising_autoencoder import ConvolutionalDenoisingAutoencoder

model = ConvolutionalDenoisingAutoencoder(in_channels=1, latent_dim=128)
```
