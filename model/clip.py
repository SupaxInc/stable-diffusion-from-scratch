import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

"""
IMPORTANT: Only created the text encoder of the CLIP encoder. The image encoder will be handled differently.
The architecture is very similar to the original Transformers architecture.
"""

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_position_embeddings: int):
        super().__init__()
        """
        Creating the input layer for the CLIP encoder. Transforms raw token inputs into embeddings.
        Preparing the data for processing through the subsequent CLIP layers.

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
        # Helps add more comple positional relationships in the input sequences in CLIP
        x += self.position_embedding

        # (Batch, Seq_Len, Dim)
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int):
        """
        Create a CLIP layer that processes and transforms input embeddings.

        Args:
            n_heads: Number of attention heads for the self-attention mechanism.
            embed_dim: Dimensionality of the input embeddings.

        The CLIP layer's purpose is to:
        1. Capture complex relationships between different parts of the input (text or image).
        2. Transform and refine the input representations through multiple such layers.
        3. Generate rich, context-aware embeddings that can be used for various downstream tasks,
           particularly for aligning text and image representations in a shared embedding space.
        """
        super().__init__()

        # Layer normalization before self-attention for stable learning
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        
        # Self-attention mechanism
        self.attention = SelfAttention(n_heads, embed_dim)
        
        # Layer normalization before feed-forward network
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network:
        # First layer expands dimensionality with a scale of 4 to allow the network to learn richer representations
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        # Activation function for non-linearity
        self.activation = lambda x: x * torch.sigmoid(1.702 * x)  # QuickGELU activation function for faster computation than normal GELU
        # Second layer projects back to original dimensionality, this compression helps in extracting the learned features
        self.linear_2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The layer performs the following operations:
        1. Self-attention: Allows the model to weigh the importance of different parts of the input sequence.
        2. Feed-forward network: Further processes the attention output, allowing for more complex transformations.
        Both operations are wrapped with residual connections and layer normalizations, which help in training stability and information flow.

        Args:
            x: Input vector representation embeddings that was outputted from the CLIP Embedding process (Batch, Seq_Len, Dim).
        
        Returns:
            torch.Tensor: Embedding that represents the semantic meaning of the input text (Batch, Seq_Len, Dim).
        """

        # Self attention layer
        residue = x  # (Batch, Seq_Len, Dim)

        # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim)
        x = self.layer_norm_1(x)

        # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)

        # (Batch, Seq_Len, Dim) + (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim)
        x += residue

        # Feed forward network layer
        residue = x  # (Batch, Seq_Len, Dim)

        # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim)
        x = self.layer_norm_2(x)

        # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, 4*Dim)
        x = self.linear_1(x)

        # (Batch, Seq_Len, 4*Dim) -> (Batch, Seq_Len, 4*Dim)
        x = self.activation(x)

        # (Batch, Seq_Len, 4*Dim) -> (Batch, Seq_Len, Dim)
        x = self.linear_2(x)

        # (Batch, Seq_Len, Dim) + (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim)
        x += residue

        # (Batch, Seq_Len, Dim)
        return x

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