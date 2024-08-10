import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        """
        Args:
            n_heads: Number of attention heads.
            d_embed: Dimension of the embedding (features/channels) for each pixel.
            in_proj_bias: Adds a bias term to the input projection.
            out_proj_bias: Adds a bias term to the output projection.
        """
        super().__init__()

        # This linear layer represents the WQ, WK, WV parameter matrices combined
        # It transforms the input embedding into Q', K', V' vectors
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # This represents the WO matrix used to transform the concatenated attention outputs
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        # Number of attention heads for focusing on different parts of the input
        # This represents the multiple heads split from the vectors (Q1, Q2... K1, K2.. V1, V2, etc)
        self.n_heads = n_heads
        # Dimension of each attention head (d_k in the formula of each head after ebing split from vectors)
        self.d_head = d_embed // n_heads 

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        """
        Self attention helps the model consider the relationships between different pixels (tokens) as each pixel has its 
        own embedding (the features) which help capture long-range dependencies and enhances the contextual understanding of the image.
        
        Args:
            x: Input tensor of shape (Batch, Sequence Length, d_embed (Dim)).
            causal_mask: Applies a causal mask to prevent attending to future pixels. 
                         This means that each pixel can only consider previous and current pixels, not future ones.

        Returns:
            torch.Tensor: Output tensor after self-attention, shape (batch_size, seq_len, d_embed).

        Process:
        1. Project input to query, key, and value vectors.
        2. Split vectors into multiple heads.
        3. Compute attention scores between query and key.
        4. Apply causal mask if specified.
        5. Compute attention weights using softmax.
        6. Apply attention weights to values vector (V').
        7. Concatenate multi-head outputs.
        8. Project concatenated output to final dimension.
        """

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # 1: Project input to Q', K', V' vectors, essentially multiplying input tensor with combined WQ, WV, WK matrices
            # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim * 3) -> chunk to 3 tensors of shape (Batch, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # 2: Split the vectors into the number of heads Q1, Q2... K1, K2.. V1, V2, etc
            # Each (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Num Heads, Dim / Num Heads) -> (Batch, Num Heads, Seq_Len, Dim / Num Heads)
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        # Now work on computing the attention using the formula: Attention(Q, K, V) = softmax((Q * KT) / √dk) * V

        # Q * KT
        # 3: Compute attention scores between query and key by: query matrix multiplication by transposed of the keys
            # Effectively calculates a score that represents the similarity between each pair of pixels
            # (Batch, Num Heads, Seq_Len, Dim / Num Heads) @ (Batch, Num Heads, Dim / Num Heads, Seq_Len) -> (Batch, Num Heads, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        # 4: Apply causal mask (optional), to prevent pixels from attending to future positions
        if causal_mask:
            # Create a mask where all elements above the main diagonal are set to 1 (True)
            # This means for a matrix any element i,j (row, column) where i < j will be a 1 (True)
            # Effectively masking future positions in a sequence of pixels, ensuring each position only considers past and present info
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            
            # Apply the mask to the weight tensor
            # For each True value in the mask, replace the corresponding element in weight with -infinity
            # This effectively prevents attention to future positions by setting their scores to -infinity
            # After softmax, these -infinity values will become 0, ensuring no attention is paid to future positions
            weight.masked_fill_(mask, -torch.inf)

        # (Q * KT) / √dk
        # Scaling helps to prevent the dot product values from becoming too large, which can lead to very small gradients during training
        weight /= math.sqrt(self.d_head)

        # softmax((Q * KT) / √dk)
        # 5: Compute the current calculated attention weights to softmax
            # (Batch, Num Heads, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) # Apply softmax along the last dimension (seq len)

        # 6: Applying attention weights to values vector (V')
            # (Batch, Num Heads, Seq_Len, Seq_Len) @ (Batch, Num Heads, Seq_Len, Dim / Num Heads) -> (Batch, Num Heads, Seq_Len, Dim / Num Heads)
        output = weight @ v

        # Prepare for concatenation by moving sequence length before number of heads
            # (Batch, Num Heads, Seq_Len, Dim / Num Heads) -> (Batch, Seq_Len, Num Heads, Dim / Num Heads)
        output = output.transpose(1,2)

        # 7: Concatenating the multi-head attention results (the H in the formula with shape seq, H * dk)
        # Where the concatenation occurs. Reshaping and keeping tensor data to combine results from all heads.
        # Flattens the last two dimensions (Num_Heads and Dim / Num_Heads) into a single dimension (Dim) effectively concatenating all heads for each position.
            # (Batch, Seq_Len, Num Heads, Dim / Num Heads) -> (Batch, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # 8. Project concatenated output to final dimension. Transforming the concatenated outputs with matrix multiplication with WO to get MH-A result
            # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim)
        output = self.out_proj(output)

        # (Batch, Seq_Len, Dim (d_embed))
        return output
