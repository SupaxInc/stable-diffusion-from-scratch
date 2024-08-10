import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_position_embeddings: int):
        super().__init__()
        """
        Args:
            vocab_size: The size of the vocabulary. This determines the number of unique tokens
                        that can be represented.

            embed_dim: The dimensionality of the token embeddings. This is the size of the vector
                        that represents each token.

            max_position_embeddings: The maximum number of positions that can be embedded. This 
                                     limits the maximum length of the input sequence.

        The embeddings token, and position work together to provide a representation of the input text,
        capturing both semantic meaning and positional information.
        """
        # Convert each token to a dense vector representation
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Encode the position of each token in the sequence.
        self.position_embedding = nn.Parameter(torch.zeros(max_position_embeddings, embed_dim))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            tokens: Input tensor of token indices (Batch, Seq_Len).

        Returns:
            torch.FloatTensor: Embedded representation of the input tokens (Batch, Seq_Len, Dim).
        """

        # (Batch, Seq_Len) -> (Batch, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        # Add the positional encoding to each embedding (it is not a fixed sinusoidal function but are learnt params from the model)
        # Helps add more comple positional relationships in the input sequences for CLIP
        x += self.position_embedding

        # (Batch, Seq_Len, Dim)
        return x

"""
    Look at the Self Attention header of StableDiffusion.md, we will build CLIP similar to the transformer image on the
    left hand side of the Self Attention picture. It begins with Positional Encoding that tells us the position of each token in a sequence
    and it is made up of multiple layers of Attention and Feed Forwards that go one after another.
"""
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # The resulting embeddings capture both semantic meaning and positional information of tokens
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
            tokens: Input tensor of token indices (Batch, Seq_Len).
                    Uses LongTensor because CLIP encoders typically work with integer token indices,
                    where each index corresponds to a word or subword in the vocabulary.
        
        Returns:
            torch.FloatTensor: Encoded representation of the input tokens (Batch, Seq_Len, Dim).
        """
        tokens = tokens.type(torch.long)

        # Convert token indices to embeddings
        # (Batch, Seq_Len) -> (Batch, Seq_Len, Dim)
        state = self.embeddings(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layer_norm(state)

        # Return the encoded representation
        # (Batch, Seq_Len, Dim)
        return output