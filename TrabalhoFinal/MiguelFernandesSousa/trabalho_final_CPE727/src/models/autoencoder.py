"""
Autoencoder models for unsupervised pre-training and denoising.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for spectrogram features.
    
    Can be used for:
    - Unsupervised pre-training
    - Dimensionality reduction
    - Feature learning
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 64,
        encoder_channels: List[int] = [32, 64, 128],
    ):
        """
        Initialize Convolutional Autoencoder.
        
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
            encoder_channels: Channels for encoder layers
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_ch = input_channels
        
        for out_ch in encoder_channels:
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Bottleneck (will be initialized dynamically)
        self.flatten = nn.Flatten()
        self.fc_encode = None
        self.fc_decode = None
        
        # Decoder (reverse of encoder)
        decoder_layers = []
        decoder_channels = list(reversed(encoder_channels))
        
        for i, out_ch in enumerate(decoder_channels[:-1]):
            in_ch = decoder_channels[i]
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
        
        # Final layer
        decoder_layers.append(
            nn.ConvTranspose2d(
                decoder_channels[-1],
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
        )
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.encoded_shape = None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        x = self.encoder(x)
        
        # Save shape for decoding
        if self.encoded_shape is None:
            self.encoded_shape = x.shape[1:]
        
        # Flatten and project to latent space
        x = self.flatten(x)
        
        if self.fc_encode is None:
            self.fc_encode = nn.Linear(x.size(1), self.latent_dim).to(x.device)
            self.fc_decode = nn.Linear(
                self.latent_dim,
                int(torch.prod(torch.tensor(self.encoded_shape)))
            ).to(x.device)
        
        x = self.fc_encode(x)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        x = self.fc_decode(z)
        x = x.view(-1, *self.encoded_shape)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed


class DenoisingAutoencoder(ConvolutionalAutoencoder):
    """
    Denoising Autoencoder that adds noise during training.
    
    Useful for learning robust features.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 64,
        encoder_channels: List[int] = [32, 64, 128],
        noise_factor: float = 0.3,
    ):
        """
        Initialize Denoising Autoencoder.
        
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
            encoder_channels: Channels for encoder layers
            noise_factor: Factor for adding noise
        """
        super().__init__(input_channels, latent_dim, encoder_channels)
        self.noise_factor = noise_factor
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input."""
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with noise.
        
        Returns:
            Tuple of (noisy_input, reconstructed)
        """
        x_noisy = self.add_noise(x)
        z = self.encode(x_noisy)
        x_reconstructed = self.decode(z)
        return x_noisy, x_reconstructed


class SparseAutoencoder(ConvolutionalAutoencoder):
    """
    Sparse Autoencoder with L1 regularization on latent space.
    
    Encourages sparse representations.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 64,
        encoder_channels: List[int] = [32, 64, 128],
        sparsity_weight: float = 0.001,
    ):
        """
        Initialize Sparse Autoencoder.
        
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
            encoder_channels: Channels for encoder layers
            sparsity_weight: Weight for sparsity loss
        """
        super().__init__(input_channels, latent_dim, encoder_channels)
        self.sparsity_weight = sparsity_weight
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparsity.
        
        Returns:
            Tuple of (reconstructed, latent) for computing sparsity loss
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_reconstructed: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with sparsity penalty.
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed output
            z: Latent representation
            
        Returns:
            Total loss
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_reconstructed, x)
        
        # Sparsity loss (L1 on latent)
        sparsity_loss = torch.mean(torch.abs(z))
        
        # Total loss
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        
        return total_loss
