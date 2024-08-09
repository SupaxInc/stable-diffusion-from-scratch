import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

"""
    Look at the Self Attention header of StableDiffusion.md, we will build CLIP similar to the transformer image on the
    left hand side of the Self Attention picture. It begins with Positional Encoding that tells us the position of each token in a sequence
    and it is made up of multiple layers of Attention and Feed Forwards that go one after another.
"""

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch, Seq_Len) -> (Batch, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        # (Batch, Seq_Len, Dim)
        output = self.layer_norm(state)

        return output