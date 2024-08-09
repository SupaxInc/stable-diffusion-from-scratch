import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# Inheriting from Sequential sequences other modules in the order that they are added.
# The forward method applies each module in sequence
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        # Begin with initial image and keep decreasing its resolution but increasing the features
        super().__init__(
            # Initial Image (Batch, Channel, Height, Width) -> (Batch, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (Batch, 128, Height, Width) -> (Batch, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            # (Batch, 128, Height, Width) -> (Batch, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            # (Batch, 128, Height, Width) -> (Batch, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (Batch, 128, Height/2, Width/2) -> (Batch, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),
            # (Batch, 256, Height/2, Width/2) -> (Batch, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),
            # (Batch, 256, Height/2, Width/2) -> (Batch, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (Batch, 256, Height/4, Width/4) -> (Batch, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),
            # (Batch, 512, Height/4, Width/4) -> (Batch, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # Runs self attention over each pixel (relates each pixels to each other in context of images)
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),

            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            nn.SiLU(), # Similar to ReLU, this just works better in this particular application
            # (Batch, 512, Height/8, Width/8) -> (Batch, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (Batch, 8, Height/8, Width/8) -> (Batch, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Sample from the latent space of the encoder.
        
        Args: 
            x:     Input tensor to the network (Batch, Channel, Height, Width)
            noise: Sample noise added to the network (Batch, Channel, Height/8, Width/8)

        Returns:
            torch.Tensor: Output tensor shape (Batch, 4, Height/8, Width/8).
        """

        # Run each of the modules sequentially
        for module in self:
            # Checks if a module has a stride of 2x2 (stride = 2, meaning both height and width are downsampled by a factor of 2)
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_Left, Padding_Right + 1, Padding_Top, Padding_Bottom + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x) # Apply the modules/convolutions
        
        # At this point, x shape is (Batch, 8, Height/8, Width/8) due to output after sequential convolutions
        
        # Latent space of the VAE are parameters of a joint distribution from a dataset and the outputs are the mean and variance
        # (Batch, 8, Height/8, Width/8) -> two tensors of shape (Batch, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1) # Splits the input tensor into 2 chunks

        # (Batch, 4, Height/8, Width/8) -> (Batch, 4, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp() # Exponentiate the log variance to get the variance

        # Standard deviation is the sqrt of the variance
        # (Batch, 4, Height/8, Width/8) -> (Batch, 4, Height/8, Width/8)
        stdev = variance.sqrt()

        # Now sample from the latent space joint distribution
        # Transform distribution Z to N that has the mean and variance (meaning sample from the distribution)
        # Z = N(0, 1) -> N(mean, variance) = X
        # Formula to transform normal variable to desired distribution: X = mean + stdev * z
        # noise shape is (Batch, 4, Height/8, Width/8)
        # (Batch, 4, Height/8, Width/8) -> (Batch, 4, Height/8, Width/8)
        x = mean + stdev * noise

        # Scale the latent representation from: https://github.com/huggingface/diffusers/issues/437
        # The constant 0.18215 is used to normalize the range of values in the latent space
        # Ensuring that the decoder can properly interpret and reconstruct the image from this scaled latent space
        # TODO: Add this constant to a config yaml file and instantiate the value through constructor
        x *= 0.18215

        # (Batch, 4, Height/8, Width/8)
        return x