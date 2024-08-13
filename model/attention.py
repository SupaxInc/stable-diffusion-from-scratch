import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        """
        Self attention helps the model consider the relationships between different pixels (or tokens) as each pixel has its 
        own embedding (the features) which help capture long-range dependencies and enhances the contextual understanding of the image.
        
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
        # Dimension of each attention head (d_k in the formula of each head after being split from vectors)
        self.d_head = d_embed // n_heads

        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, d_embed (Dim)).
            causal_mask: Applies a causal mask to prevent attending to future pixels. 
                         This means that each pixel can only consider previous and current pixels, not future ones.

        Returns:
            torch.Tensor: Output tensor after self-attention, shape (Batch, Seq_Len, d_embed (Dim)).

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

        # New shape to transform the query, key, and values
        iterim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # 1: Project input to Q', K', V' vectors, essentially multiplying input tensor with combined WQ, WV, WK matrices
            # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim * 3) -> chunk to 3 tensors of shape (Batch, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # 2: Split the vectors into the number of heads Q1, Q2... K1, K2.. V1, V2, etc
            # Each (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Num Heads, Dim / Num Heads) -> (Batch, Num Heads, Seq_Len, Dim / Num Heads)
        q = q.view(iterim_shape).transpose(1, 2)
        k = k.view(iterim_shape).transpose(1, 2)
        v = v.view(iterim_shape).transpose(1, 2)

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

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        """
        This module implements cross-attention, allowing the model to attend to a different input
        (context) than the one being transformed. It's crucial for incorporating text embeddings
        into the image generation process.

        Args:
            n_heads: Number of attention heads.
            d_embed: Dimensionality of the input embeddings (query).
            d_cross: Dimensionality of the cross-attention input (key/value).
            in_proj_bias: Whether to include bias in input projections. Defaults to True.
            out_proj_bias: Whether to include bias in output projection. Defaults to True.
        """
        super().__init__()
        
        # Separate projection layers for query, key, and value instead of 1 big matrix combining all
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        # Output projection (WO matrix)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        # Number of attention heads for focusing on different parts of the input
        # This represents the multiple heads split from the vectors (Q1, Q2... K1, K2.. V1, V2, etc)
        self.n_heads = n_heads
        # Dimension of each attention head (d_k in the formula of each head after being split from vectors)
        self.d_head = d_embed // n_heads

        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        In cross attention, the query comes from one source (x) while the key and value come from another (y).
        This differs from self-attention where Q, K, and V all come from the same source.

        Args:
            x: Latent representation of the noisy image (Batch, Seq_Len_Q, Dim_Q).
            y: Context tensor, typically CLIP text embeddings (Batch, Seq_Len_KV, Dim_KV) = (Batch, 77, 768)

        Returns:
            torch.Tensor: Attended output (Batch, Seq_Len_Q, Dim_Q)

        Process:
            1. Project inputs to query, key, and value vectors.
            2. Split vectors into multiple heads.
            3. Compute attention scores between query and key.
            4. Scale attention scores.
            5. Compute attention weights using softmax.
            6. Apply attention weights to values vector (V').
            7. Concatenate multi-head outputs.
            8. Project concatenated output to final dimension.
        """

        # Save input shape for later reshaping
        input_shape = x.shape
        batch, seq_len, d_embed = input_shape

        # 1. Project inputs to query (WQ), key(WK), and value(WV) matrices
        # Note: Unlike self-attention where its combined into 1 matrix, q comes from x, while k and v come from y
        q = self.q_proj(x)  # (Batch, Seq_Len_Q, Dim_Q)
        k = self.k_proj(y)  # (Batch, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)  # (Batch, Seq_Len_KV, Dim_Q)

        # 2. Reshape q, k, v for multi-head attention
        # -1 is a placeholder for Seq_len so it can work with varying sequence lengths
        # Splitting the vectors into the number of heads Q1, Q2... K1, K2.. V1, V2, etc by using d_head from interim shape
        # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Num_Heads, Dim / Num_Heads) -> (Batch, Num_Heads, Seq_Len, Dim / Num_Heads)
        q = q.view((batch, -1, self.n_heads, self.d_head)).transpose(1, 2)
        k = k.view((batch, -1, self.n_heads, self.d_head)).transpose(1, 2)
        v = v.view((batch, -1, self.n_heads, self.d_head)).transpose(1, 2)

        # 3. Compute attention scores
        # (Batch, Num_Heads, Seq_Len_Q, Dim / Num_Heads) @ (Batch, Num_Heads, Dim / Num_Heads, Seq_Len_KV)
        # -> (Batch, Num_Heads, Seq_Len_Q, Seq_Len_KV)
        attention_scores = q @ k.transpose(-1, -2)

        # 4. Scale attention scores
        # Scaling helps to prevent the dot product values from becoming too large which can lead to very small gradients during training
        attention_scores /= math.sqrt(self.d_head)

        # 5. Apply softmax to get attention weights
        # Note: Unlike self-attention, no causal mask is applied here
        # Each query position can attend to all key positions including future positions
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 6. Apply attention weights to values
        # (Batch, Num_Heads, Seq_Len_Q, Seq_Len_KV) @ (Batch, Num_Heads, Seq_Len_KV, Dim / Num_Heads)
        # -> (Batch, Num_Heads, Seq_Len_Q, Dim / Num_Heads)
        output = attention_weights @ v

        # 7. Reshape output and concatenate heads
        # Reshape to match the query's sequence length (Seq_Len_Q) b/c the output should have the same sequence length as the input query (x)
        # (Batch, Num_Heads, Seq_Len_Q, Dim / Num_Heads) -> (Batch, Seq_Len_Q, Num_Heads, Dim / Num_Heads)
        output = output.transpose(1, 2).contiguous()
        # (Batch, Seq_Len_Q, Num_Heads, Dim / Num_Heads) -> (Batch, Seq_Len_Q, Dim)
        output = output.view(input_shape)

        # 8. Apply output projection
        output = self.out_proj(output)

        # The output maintains the shape (Batch, Seq_Len_Q, Dim_Q) to match the input query dimensions
        return output
    