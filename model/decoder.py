import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import SelfAttention

"""
    Inheriting nn.Module provides the framework to define custom models
    We would need to define the forward method explicitly to specify how the input flows through the network
"""
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            # If the number of input channels is equal to the number of output channels, use an identity layer for the residual connection.
            # The identity layer passes the input directly to the output without any changes.
            self.residual_layer = nn.Identity()
        else:
            # If its different, use a 1x1 convolution to match the input channel as output channel for the residual connection.
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block for a VAE.

        This method helps mitigate the vanishing gradient problem by using residual connections.
        The input tensor is then added back to the output tensor to preserve information and ensure stable gradients.

        Args: 
             x: Input tensor to the network (Batch, In_Channel, Height, Width)

        Returns:
             torch.Tensor: Output tensor after applying the residual block (Batch, Out_Channel, Height, Width)
        """

        # Residue preserves the input tensor to be added back to the output after passing through the conv layers
        residue = x 

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        # Identity layer/Conv2d on the residue helps ensure the input (residue) and output (x) tensors have the same shape
            # Allows them to be added together
            # Helps mitigate the vanishing gradient problem
        return x + self.residual_layer(residue)

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
             x: Input tensor to the network (Batch, Features, Height, Width)
        """

        residue = x
        b, c, h, w = x.shape

        # Do the self attention between all the pixels of the image

        # Reshape the tensor without changing its data, preparing the tensor for attention mechanisms
        # (Batch, Features, Height, Width) -> (Batch, Features, Height * Width)
        x = x.view(b, c, h * w)

        # Tranposing is necessary for attention as each pixel has its own embedding which are the features 
        # and it helps relates the pixels to each other
        # (Batch, Features, Height * Width) -> (Batch, Height * Width, Features)
        x = x.transpose(-1, -2)

        # (Batch, Height * Width, Features) -> (Batch, Height * Width, Features)
        x = self.attention(x)

        # (Batch, Height * Width, Features) -> (Batch, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch, Features, Height * Width) -> (Batch, Features, Height, Width)
        x = x.view(b, c, h, w)

        return x + residue

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Initial processing of latent representation
            # (Batch, 4, Height/8, Width/8) -> (Batch, 4, Height/8, Width/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # Expand channels for feature extraction
            # (Batch, 4, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # Series of residual blocks for deep feature processing, ensuring stable gradients
            # Maintain shape: (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # Apply self-attention to capture global context
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),
            
            # Additional residual blocks for further processing
            # All maintain shape: (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # Begin upsampling process
            # Upsampling increases the spatial dimensions of the feature maps
            # This step is crucial in the decoder to gradually restore the original image size
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/4, Width/4)
            nn.Upsample(scale_factor=2),

            # Maintain channel size after upsampling and process the expanded spacial dimensions
            # (Batch, 512, Height/4, Width/4) -> (Batch, 512, Height/4, Width/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # All maintain shape: (Batch, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Second upsampling
            # (Batch, 512, Height/4, Width/4) -> (Batch, 512, Height/2, Width/2)
            nn.Upsample(scale_factor=2),
            # Maintain channel size after upsampling
            # (Batch, 512, Height/2, Width/2) -> (Batch, 512, Height/2, Width/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # Reduce channels and further process
            # (Batch, 512, Height/2, Width/2) -> (Batch, 256, Height/2, Width/2)
            VAE_ResidualBlock(512, 256),
            # Maintain shape: (Batch, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # Final upsampling to original image size
            # (Batch, 256, Height/2, Width/2) -> (Batch, 256, Height, Width) 
            nn.Upsample(scale_factor=2),
            # Maintain channel size after upsampling
            # (Batch, 256, Height, Width) -> (Batch, 256, Height, Width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # Further reduce channels and process
            # (Batch, 256, Height, Width) -> (Batch, 128, Height, Width)
            VAE_ResidualBlock(256, 128),
            # Maintain shape: (Batch, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # Final normalization and activation
            # (Batch, 128, Height, Width) -> (Batch, 128, Height, Width)
            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # Output layer: reduce to 3 channels for RGB image
            # (Batch, 128, Height, Width) -> (Batch, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x: Input tensor to the network which is the output of the encoder (Batch, Features, Height/8, Width/8)
        
        Returns:
            torch.Tensor: Output tensor shape (Batch, 3 Channels, Height, Width).
        """

        # Reverse the scaling of the output of the sampling done from the encoder
        # TODO: Add this constant to a config yaml file and instantiate the value through constructor
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch, 3, Height, Width), now an RGB image
        return x
        
