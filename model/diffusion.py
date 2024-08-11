import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        # Feed forward network
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, 4 * embed_dim) # Larger scaling factor than CLIP encoder to capture more complex temporal info
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input time step tensor (1, 320).
        
        Returns:
            torch.Tensor: Timestep tensor converted to a time embedding (1, 1280)
        """

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)
        
        # (1, 1280)
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