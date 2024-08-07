import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        """
        Args:
            n_heads (int): Number of attention heads.
            d_embed (int): Dimension of the embedding (features/channels) for each pixel.
            in_proj_bias (bool, optional): Adds a bias term to the input projection.
            out_proj_bias (bool, optional): Adds a bias term to the output projection.
        """
        super().__init__()

        # Defining the three WQ, WV, WV parameter matrices as one big linear layer
            # Each three heads have dimensions of d_model x d_model
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # Defining the WO matrix (output matrix) used to multiply with the concatenated heads
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        # Number of outputted heads to help focus on different parts of the image
        self.n_heads = n_heads
        # Each head will watch a part of the embedding of each pixel
        self.d_head = d_embed // n_heads 
