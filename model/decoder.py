import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

# Inheriting nn.Module provides the framework to define custom models
# We would need to define the forward method explicitly to specify how the input flows through the network
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.groupNorm = nn.GroupNorm(32, channels)


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupNorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupNorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            # If the number of input channels is equal to the number of output channels, use an identity layer for the residual connection.
            # The identity layer passes the input directly to the output without any changes.
            self.residual_layer = nn.Identity()
        else:
            # If its different, use a 1x1 convolution to match the input channel as output channel for the residual connection.
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
             x: Input tensor to the network (Batch, In_Channel, Height, Width)
        """

        # Residue preserves the input tensor to be added back to the output after passing through the conv layers
        residue = x 

        x = self.groupNorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupNorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        # Identity layer/Conv2d on the residue helps ensure the input (residue) and output (x) tensors have the same shape
            # Allows them to be added together
            # Helps mitigate the vanishing gradient problem
        return x + self.residual_layer(residue)
