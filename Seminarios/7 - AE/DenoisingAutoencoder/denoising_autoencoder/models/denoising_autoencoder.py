import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder (DAE) base class.

    This class serves as a template for building denoising autoencoders.
    It provides the basic structure and common methods for encoding and decoding.

    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        latent_dim (int): Dimension of the latent space representation
    """

    def __init__(self, in_channels=1, latent_dim=128):
        super(DenoisingAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

    def encode(self, x):
        """Encode input to latent representation."""
        raise NotImplementedError("Encode method not implemented.")

    def decode(self, z):
        """Decode latent representation to reconstructed image."""
        raise NotImplementedError("Decode method not implemented.")

    def forward(self, x):
        """Forward pass through the autoencoder."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed

def convBnReluMax(in_channels, out_channels, kernel_size, stride, padding, pool_kernel, pool_stride):
    """Helper function to create a Conv-BatchNorm-ReLU-MaxPool block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool_kernel, pool_stride)
    )
def convBnRelu(in_channels, out_channels, kernel_size, stride, padding):
    """Helper function to create a Conv-BatchNorm-ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def convTransposeBnRelu(in_channels, out_channels, kernel_size, stride, padding, output_padding):
    """Helper function to create a ConvTranspose-BatchNorm-ReLU block."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class ConvolutionalDenoisingAutoencoder(nn.Module):
    """
    Convolutional Denoising Autoencoder (CDAE) for image denoising.

    Architecture:
    - Encoder: Conv2d -> ReLU -> MaxPool2d (repeated)
    - Decoder: ConvTranspose2d -> ReLU (repeated)

    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        latent_dim (int): Dimension of the latent space representation
        image_size (int): Size of the input images (assumes square images)
    """

    def __init__(self, in_channels=1, latent_dim=128, image_size=28, kernel_size=3, stride=1, padding=1, out_channels=[32, 64, 128]):
        super(ConvolutionalDenoisingAutoencoder, self).__init__()

        self.in_channels = [in_channels] + out_channels[:-1]
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.out_channels = out_channels



        # Encoder
        self.encoder = nn.Sequential(*(
            convBnReluMax(in_ch, out, kernel_size, stride, padding, 2, 2) for in_ch, out in zip(self.in_channels, self.out_channels)
        ))

        # Calculate the flattened size after encoder
        self.encoded_size = 128 * (image_size // 8) * (image_size // 8)

        # Bottleneck
        self.fc_encode = nn.Linear(self.encoded_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.encoded_size)

        # Decoder - mirrors encoder with transposed convolutions
        # Use stride=2 transpose convolutions with output_padding=1 for upsampling
        # This approximately doubles dimensions, then adjust final size with interpolation or padding
        transpose_stride = 2

        decoder_channels = list(reversed(self.out_channels)) + [32,]

        # Build decoder layers
        decoder_layers = []
        for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:]):
            decoder_layers.append(
                convTransposeBnRelu(in_ch, out_ch, kernel_size=kernel_size, stride=transpose_stride, padding=padding, output_padding=1)
            )

        # Add final convolution to get correct number of channels
        decoder_layers.append(nn.Conv2d(32, in_channels, kernel_size=kernel_size, stride=1, padding=padding))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode input to latent representation."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encode(x)
        return x

    def decode(self, z):
        """Decode latent representation to reconstructed image."""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, self.image_size // 8, self.image_size // 8)
        x = self.decoder(x)

        # Ensure output matches exact image size (handle any dimension mismatches)
        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        return x

    def forward(self, x):
        """Forward pass through the autoencoder."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed
