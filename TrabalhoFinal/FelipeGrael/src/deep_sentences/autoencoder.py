#!/usr/bin/env python3
"""
Autoencoder model for NLP feature compression.
"""

from typing import List, Optional, Literal
import torch
import torch.nn as nn
import lightning as L


class Autoencoder(L.LightningModule):
    """
    Autoencoder with configurable layer dimensions and L1 regularization on latent space.

    The encoder dimensions are mirrored to create the decoder architecture.
    For example, dimensions=[2000, 500, 125, 2] creates:
    - Encoder: 2000 -> 500 -> 125 -> 2 (latent)
    - Decoder: 2 -> 125 -> 500 -> 2000 (output)
    """

    def __init__(
        self,
        dimensions: List[int],
        activation: Literal["relu", "tanh", "sigmoid", "leaky_relu"] = "relu",
        latent_activation: Literal["linear", "relu", "tanh", "sigmoid"] = "linear",
        l1_alpha: Optional[float] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
    ):
        """
        Initialize the Autoencoder.

        Args:
            dimensions: List of layer dimensions. First element is input dimension,
                       last element is latent dimension. Decoder mirrors the encoder.
            activation: Activation function for intermediate layers in encoder and decoder.
                       Options: "relu", "tanh", "sigmoid", "leaky_relu". Default: "relu"
            latent_activation: Activation function for latent space.
                             Options: "linear", "relu", "tanh", "sigmoid"
            l1_alpha: Coefficient for L1 regularization on latent activations.
                     If None, no L1 regularization is applied.
            learning_rate: Learning rate for AdamW optimizer. Default: 3e-4
            weight_decay: Weight decay (L2 regularization) for AdamW. Default: 0.01
        """
        super().__init__()
        self.save_hyperparameters()

        self.dimensions = dimensions
        self.activation = activation
        self.latent_activation = latent_activation
        self.l1_alpha = l1_alpha
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Build encoder
        encoder_layers = []
        for i in range(len(dimensions) - 1):
            encoder_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            # Apply activation to all layers except the last (latent) layer
            if i < len(dimensions) - 2:
                encoder_layers.append(self._get_activation(activation))

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent activation
        self.latent_act = self._get_activation(latent_activation)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        decoder_dims = dimensions[::-1]  # Reverse the dimensions
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            # Apply activation to all layers except the last (output) layer
            if i < len(decoder_dims) - 2:
                decoder_layers.append(self._get_activation(activation))

        self.decoder = nn.Sequential(*decoder_layers)

        # Store latent representations for L1 regularization
        self.latent = None

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "linear": nn.Identity(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        # Encode
        self.latent = self.encoder(x)
        # Apply latent activation
        latent_activated = self.latent_act(self.latent)
        # Decode
        reconstructed = self.decoder(latent_activated)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        latent = self.encoder(x)
        latent_activated = self.latent_act(latent)
        return latent_activated

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        return self.decoder(z)

    def compute_loss(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with optional L1 regularization on latent space.

        Args:
            x: Original input
            x_reconstructed: Reconstructed output

        Returns:
            Total loss (reconstruction + L1 regularization if enabled)
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x)

        # L1 regularization on latent space
        if self.l1_alpha is not None and self.latent is not None:
            l1_loss = self.l1_alpha * torch.mean(torch.abs(self.latent))
            total_loss = reconstruction_loss + l1_loss
            return total_loss, reconstruction_loss, l1_loss
        else:
            return reconstruction_loss, reconstruction_loss, torch.tensor(0.0, device=x.device)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, _ = batch  # Assuming batch is (features, targets)
        x_reconstructed = self(x)
        total_loss, recon_loss, l1_loss = self.compute_loss(x, x_reconstructed)

        # Log metrics
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss)
        if self.l1_alpha is not None:
            self.log("train_l1_loss", l1_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, _ = batch
        x_reconstructed = self(x)
        total_loss, recon_loss, l1_loss = self.compute_loss(x, x_reconstructed)

        # Log metrics
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss)
        if self.l1_alpha is not None:
            self.log("val_l1_loss", l1_loss)

        return total_loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with ReduceLROnPlateau scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # ReduceLROnPlateau: automatically reduces LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Reduce LR by 50%
            patience=5,  # Wait 5 epochs before reducing
            min_lr=self.learning_rate * 0.01,  # Don't go below 1% of initial LR
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitor validation loss
                "interval": "epoch",
                "frequency": 1,
            }
        }
