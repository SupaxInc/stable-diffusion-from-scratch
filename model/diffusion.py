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
        # Expand dimensionality
        x = self.linear_1(t)
        
        # Non-linear activation
        x = self.activation(x)
        
        # Further transform
        x = self.linear_2(x)
        
        # (Batch, 4*embed_dim)
        return x

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