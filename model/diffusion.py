import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.activation = nn.SiLU()  # SiLU activation (also known as Swish)

        # Feed forward network
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, 4 * embed_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Convert a scalar timestep that is used for the generated noisy latent into a feature vector (time embedding).

        In diffusion models, each noise level (timestep) needs to be represented
        as a vector to provide temporal information to the model. This embedding
        allows the model to understand and differentiate between different stages
        of the diffusion process.

        Args:
            t: Input timestep tensor (1, 320).
        
        Returns:
            torch.Tensor: Time embedding (Batch, 4*embed_dim)
        """
        # Expand dimensionality of the time step
        # (1, 320) -> (1, 1280)
        x = self.linear_1(t)
        
        # Non-linear activation
        # (1, 1280) -> (1, 1280)
        x = self.activation(x)
        
        # Further transform
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)
        
        # (1, 1280)
        return x
    
class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        A custom Sequential module to allow flexibility of handling different types of layers in a U-Net architecture.
        The UNet will have multiple inputs to guide the diffusion model to the correct image. We need a way to process these
        different inputs for every layer.

        Args:
            x: Latent representation at the current noise level (Batch, Channels, Height, Width).
            context: Text embedding from the CLIP text encoder (Batch, Seq_Len, Dim).
            time_embedding: Embedding of the current timestep (Batch, Time_Dim).

        Returns:
            torch.Tensor: Processed tensor with the same shape as the input x (Batch, Channels, Height, Width).
        """

        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                # Attention blocks use the context for cross-attention
                # This allows the model to incorporate text information into the image generation process
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                # Residual blocks incorporate time information
                # Helps model understand the noise level at each step of the diffusion process, allowing it to gradually denoise the image
                x = layer(x, time_embedding)
            else:
                # Other layers (e.g., Conv2d) only process x
                x = layer(x)
        
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder path
        self.encoders = nn.Module([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            
        ])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        # Used to understand the timestep that was used to create the noisy image
        self.time_embedding = TimeEmbedding(320)

        self.unet = UNet()

        self.final = UNet_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: The noisy latent image (variable Z) sampled from the VAE encoder (Batch, 4, Height/8, Width/8).
            context: Embeddings that capture the semantic relationships of the text prompt (Batch, Seq_Len, Dim).
            timestep: The timestep that was used to generate the noisy latent image. (1, 320).
        """

        # (1, 320) -> (1, 1280)
        timestep = self.time_embedding(timestep)

        # (Batch, 4, Height/8, Width/8) -> (Batch, 320, Height/8, Width/8)
        output = self.unet(latent, context, timestep)

        # (Batch, 320, Height/8, Width/8) -> (Batch, 4, Height/8, Width/8)
        output = self.final(output)

        # (Batch, 4, Height/8, Width/8)
        return output