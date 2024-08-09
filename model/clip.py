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
        super().__init__()
        # 1. CLIP embeddings convert tokens (which are already numbers) into dense vector representations
        # 2. Each token is represented by a vector of size 768 (compared to 512 in original transformers)
        # 3. The embedding layer combines token embeddings and position embeddings
        # 4. 49408 is the vocabulary size, 768 is the embedding dimension, and 77 is the maximum sequence length
        # 5. The resulting embeddings capture both semantic meaning and positional information of tokens
        # TODO: Move these values to a config yaml file in the future
        self.embeddings = CLIPEmbedding(vocab_size=49408, embed_dim=768, max_position_embeddings=77)

        # 12 layers of self-attention and feed-forward, used for processing and encoding the tokens (Nx from formula)
        self.layers = nn.ModuleList([
            CLIPLayer(768) for _ in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            tokens: Input tensor of token indices. Shape: (Batch, Seq_Len).
                    Uses LongTensor because CLIP encoders typically work with integer token indices,
                    where each index corresponds to a word or subword in the vocabulary.
        
        Returns:
            torch.FloatTensor: Encoded representation of the input tokens. Shape: (Batch, Seq_Len, Dim).
        """
        tokens = tokens.type(torch.long)

        # Convert token indices to embeddings
        # (Batch, Seq_Len) -> (Batch, Seq_Len, Dim)
        state = self.embeddings(tokens)

        # Apply multiple layers of self-attention and feed-forward networks
        for layer in self.layers:
            state = layer(state)
        
        output = self.layer_norm(state)

        # Return the encoded representation
        # (Batch, Seq_Len, Dim)
        return output