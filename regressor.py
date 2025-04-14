import torch
from torch import nn

class Regressor(nn.Module):
    """
    Feed-forward neural network for regression tasks on top of frozen feature encodings.

    Composed of multiple linear layers with weight normalization, LayerNorm, and GELU activations.
    """
    def __init__(self, dim, num_labels=3):
        """
        Initializes the regressor module.

        Arguments:
            dim (int): Dimension of the input features.
            num_labels (int): Number of output labels for the regression task.
        """
        super(Regressor, self).__init__()

        self.bottleneck1 = nn.Linear(dim, 512)
        self.bottleneck1 = nn.utils.weight_norm(self.bottleneck1, name="weight")
        self.normalization1 = nn.LayerNorm
        self.activation1 = nn.GELU
        self.bottleneck2 = nn.Linear(512, 256)
        self.bottleneck2 = nn.utils.weight_norm(self.bottleneck2, name="weight")
        self.normalization2 = nn.LayerNorm
        self.activation2 = nn.GELU
        self.bottleneck3 = nn.Linear(256, 128)
        self.bottleneck3 = nn.utils.weight_norm(self.bottleneck3, name="weight")
        self.normalization3 = nn.LayerNorm
        self.activation3 = nn.GELU
        self.bottleneck4 = nn.Linear(128, 64)
        self.bottleneck4 = nn.utils.weight_norm(self.bottleneck4, name="weight")
        self.normalization4 = nn.LayerNorm
        self.activation4 = nn.GELU
        self.linear = nn.Linear(64, num_labels)
        self.linear = nn.utils.weight_norm(self.linear, name="weight")

        self.regressor = torch.nn.Sequential(
            self.bottleneck1,
            self.normalization1(512),
            self.activation1(),
            self.bottleneck2,
            self.normalization2(256),
            self.activation2(),
            self.bottleneck3,
            self.normalization3(128),
            self.activation3(),
            self.bottleneck4,
            self.normalization4(64),
            self.activation4(),
            self.linear
        )

    def forward(self, x):
        """
        Forward pass of the regressor.

        Arguments:
            x (Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_labels).
        """
        x = x.view(x.size(0), -1)

        return self.regressor(x)
    
