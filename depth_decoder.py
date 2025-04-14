## Based on 'Collision Aware In-Hand 6D Object Pose Estimation using vision-based tactile sensors' by Caddeo et al.
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    CNN-based decoder that reconstructs an image from a latent representation.

    Composed of a fully connected layer followed by a series of convolutional
    and upsampling layers to produce an output image.
    """

    def __init__(self,
                 image_size_w=128,
                 image_size_h=128,
                 latent_size=128,
                 filters=(128, 256, 256, 512),
                 conv=(5, 5, 5, 5),
                 stride=(2, 2, 2, 2)):
        """
        Initializes the decoder network.

        Arguments:
            image_size_w (int): Width of the output image.
            image_size_h (int): Height of the output image.
            latent_size (int): Size of the input latent vector.
            filters (tuple): Tuple specifying the number of filters in each layer.
            conv (tuple): Tuple specifying the kernel sizes for each layer.
            stride (tuple): Tuple specifying the stride values for each layer.
        """
        super().__init__()

        # Set the parameters of the network
        in_channels = list(reversed(filters))
        out_channels = in_channels[1:] + [3]  # Final layer outputs 3 channels (RGB)
        conv = list(reversed(conv))
        stride = list(reversed(stride))

        # Initialize deconvolutional layers
        self.deconvs = nn.ModuleList([
            nn.Conv2d(in_channels=ic,
                      out_channels=oc,
                      kernel_size=k,
                      padding=k // 2)
            for ic, oc, k in zip(in_channels, out_channels, conv)
        ])

        # Initialize upsampling layers
        self.ups = nn.ModuleList([
            nn.Upsample(scale_factor=s, mode='nearest') for s in stride
        ])

        # Compute total stride factor
        self.stride_factor = 1
        for s in stride:
            self.stride_factor *= s

        # Output image dimensions and latent projection
        self.image_size_w = image_size_w
        self.image_size_h = image_size_h
        self.last_filter = in_channels[0]

        output_linear_size = int(self.last_filter *
                                 (image_size_w / self.stride_factor) *
                                 (image_size_h / self.stride_factor))

        self.fc = nn.Linear(latent_size, output_linear_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Forward pass of the decoder.

        Arguments:
            x (Tensor): Input latent tensor of shape (batch_size, latent_size).

        Returns:
            Tensor: Output image tensor of shape (batch_size, 3, image_size_h, image_size_w).
        """
        x = self.fc(x)
        x = F.relu(x)
        x = x.view((-1, self.last_filter,
                    self.image_size_h // self.stride_factor,
                    self.image_size_w // self.stride_factor))

        for conv, up in zip(self.deconvs[:-1], self.ups[:-1]):
            x = up(x)
            x = conv(x)
            x = F.relu(x)

        x = self.ups[-1](x)
        x = self.deconvs[-1](x)

        return F.leaky_relu(x)