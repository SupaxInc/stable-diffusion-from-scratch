import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import SelfAttention, CrossAttention

# TODO: Create config.yaml file for default values

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        """
        Further process the initial time embedding to expand its dimensionality.
        """
        super().__init__()

        self.activation = nn.SiLU()  # SiLU activation (also known as Swish)

        # Feed forward network
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, 4 * embed_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Further process the initial time embedding to provide more detailed
        temporal information to the model. This expanded embedding allows
        the model to better understand and differentiate between different
        stages of the diffusion process.

        Args:
            t: Input time embedding tensor (1, 320).
        
        Returns:
            torch.Tensor: Expanded time embedding (1, 4*embed_dim)
        """
        # Expand dimensionality of the time embedding
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
    
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample the tensor with a scale factor of 2.

        Args:
            x: Input tensor to upsample (Batch, Features, Height, Width)
        """

        # (Batch, Features, Height, Width) -> (Batch, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # (Batch, Features, Height * 2, Width * 2)
        return self.conv(x)

class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_features=1280):
        """
        Residual blocks allows us to preserve information and ensure stable gradients. For UNet it gives us the ability
        to incorporate the the time information with the latent to help model understand the noise level at each step of the diffusion process, 
        allowing it to gradually denoise the image.

        Args:
            in_channels: Input channels of tensor that is being used.
            out_channels: Output channels to transform the tensor to.
            time_embedding_features: The features of the timestep when it has been converted to a feature vector (scale factor of 4).
        """
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.activation = nn.SiLU()
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(time_embedding_features, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            # If the number of input channels is equal to the number of output channels, use an identity layer for the residual connection.
            # The identity layer passes the input directly to the output without any changes.
            self.residual_layer = nn.Identity()
        else:
            # If its different, use a 1x1 convolution to match the input channel as output channel for the residual connection.
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, z: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent representation (z) of noise generated for current timestep from encoder (Batch, 4, Height/8, Width/8).
            time_embedding: Timestep feature vector for current noise level (Batch, Dim * 4) -> (1, 1280).
        
        Returns:
            torch.Tensor: The merged tensor of latent and time embedding (Batch, Out_Channels, Height, Width).
        """

        # Store the input for residual connection
        residue = z

        # Process the latent representation
        # (Batch, In_Channels, Height, Width) -> (Batch, In_Channels, Height, Width)
        z = self.groupnorm_feature(z)
        z = self.activation(z)
        # (Batch, In_Channels, Height, Width) -> (Batch, Out_Channels, Height, Width)
        z = self.conv_feature(z)

        # Process the time embedding
        # (Batch, 1280) -> (Batch, 1280)
        time_embedding = self.activation(time_embedding)
        # (Batch, 1280) -> (Batch, Out_Channels)
        time_embedding = self.linear_time(time_embedding)

        # Merge the latent with the time embedding
        # Adding height and width dimension to time embedding feature vector
        # (Batch, Out_Channels, Height, Width) + (Batch, Out_Channels, 1, 1) -> (Batch, Out_Channels, Height, Width)
        merged = z + time_embedding.unsqueeze(-1).unsqueeze(-1)
        
        # Further process the merged representation
        # (Batch, Out_Channels, Height, Width) -> (Batch, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        merged = self.activation(merged)
        merged = self.conv_merged(merged)
    
        # Apply residual connection
        # The residual layer adapts the input dimensions if necessary (when in_channels != out_channels)
        # (Batch, Out_Channels, Height, Width) + (Batch, Out_Channels, Height, Width) -> (Batch, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)
    
class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int, context_dim: int = 768):
        """
        This block combines self-attention and cross-attention mechanisms to process
        latent representations and incorporate context information (e.g., text embeddings).
        It's a crucial component in allowing the model to understand spatial relationships
        within the image and align them with textual descriptions.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Dimension of the model (per head).
            context_dim: Dimension of the context vector in this case the CLIPEmbedding. 
                            Defaults to 768 based on clip encoder in clip.py.

        The block consists of:
        1. Self-attention: Allows the model to relate different parts of the image to each other.
        2. Cross-attention: Enables the model to incorporate external context (e.g., text embeddings from CLIP text encoder).
        3. Feed-forward network: Further processes the attention outputs.
        """
        super().__init__()
        self.channels = n_heads * embed_dim

        # Initial normalization and projection
        self.groupnorm = nn.GroupNorm(32, self.channels, eps=1e-6)
        self.conv_input = nn.Conv2d(self.channels, self.channels, kernel_size=1, padding=0)

        # Self-attention block
        self.layernorm_1 = nn.LayerNorm(self.channels)
        self.self_attention = SelfAttention(n_heads, self.channels, in_proj_bias=False)

        # Cross-attention block
        self.layernorm_2 = nn.LayerNorm(self.channels)
        self.cross_attention = CrossAttention(n_heads, self.channels, context_dim, in_proj_bias=False)

        # Feed-forward network (FFN) block using GEGLU activation function (Gated Element-wise Linear Unit)
        self.layernorm_3 = nn.LayerNorm(self.channels)
        # 1) Expand dimensionality by factor of 4 and split into two halves:
        #    - One half for linear transformation
        #    - One half for gating mechanism
        self.linear_geglu_1 = nn.Linear(self.channels, 4 * self.channels * 2)
        # GELU activation function for non-linearity
        self.activation = nn.GELU()
        # 2) Bring the dimension back to original size
        # This compression helps in extracting the most important features learned
        self.linear_geglu_2 = nn.Linear(4 * self.channels, self.channels)

        # Output projection
        self.conv_output = nn.Conv2d(self.channels, self.channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention block imilar to a Transformer architecture.

        Args:
            x: Latent of shape (Batch, Features, Height, Width).
            context: Text embeddings (CLIPEmbedding) of shape (Batch, Seq_Len, Dim).

        Returns:
            torch.Tensor: Processed tensor (Batch, Features, Height, Width).
        """

        # Long residual connection: Preserves the original input information
        # This helps in maintaining the overall structure and low-level features of the image
        residue_long = x  # Will be added back at the very end of the block

        x = self.groupnorm(x)
        x = self.conv_input(x)

        b, c, h, w = x.shape

        # Reshape for attention operations
        # (Batch, Features, Height, Width) -> (Batch, Features, Height * Width)
        x = x.view((b, c, h*w))
        # (Batch, Features, Height * Width) -> (Batch, Height * Width, Features)
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection
        residue_short = x # Short residual: Helps in gradient flow and preserves local information

        x = self.layernorm_1(x)
        self.self_attention(x)
        x += residue_short  # Skip connection: Allows the model to bypass self-attention if necessary

        # # Normalization + Cross Attention with skip connection
        residue_short = x # Another short residual: Enables the model to selectively use text context

        x = self.layernorm_2(x)
        self.cross_attention(x, context)
        x += residue_short  # Skip connection: Model can choose to ignore text context if not relevant

        # Normalization + Feed-forward network using GEGLU with skip connection
        residue_short = x # Final short residual: Allows for complex non-linear transformations while preserving input

        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * self.activation(gate)
        x = self.linear_geglu_2(x)
        x += residue_short  # Skip connection: Enables the model to bypass FFN if simpler transformation is needed

        # Reshape back to original tensor shape
        # (Batch, Height * Width, Features) -> (Batch, Features, Height * Width)
        x = x.transpose(-1, -2)
        # (Batch, Features, Height * Width) -> (Batch, Features, Height, Width)
        x = x.view((b, c, h, w))

        # Final output with long residual connection
        # This allows the model to effectively combine the transformed features with the original input
        # Crucial for preserving spatial information and enabling fine-grained control in the diffusion process
        return self.conv_output(x) + residue_long

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        A custom Sequential module to allow flexibility of handling different types of layers in a U-Net architecture.
        The UNet will have multiple inputs to guide the diffusion model to the correct image. We need a way to process these
        different inputs for every layer.

        Args:
            x: Latent representation (z) at the current noise level (Batch, Channels/Features, Height, Width).
            context: Text embedding from the CLIP text encoder, CLIPEmbedding (Batch, Seq_Len, Dim).
            time_embedding: Embedding of the current timestep representing the current noise level (Batch, Time_Dim).

        Returns:
            torch.Tensor: Processed tensor with the same shape as the input x (Batch, Channels/Features, Height, Width).
        """

        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                # Attention blocks use the context for self and cross-attention
                # Allows the model to incorporate spatial relationships with image and text information from prompt
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                # Residual blocks incorporate time information with the noisy latent representation
                x = layer(x, time_embedding)
            else:
                # Other layers (e.g., Conv2d) only process x
                x = layer(x)
        
        return x

class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        This layer is separated from the actual UNet architecture because it's specific to the stable diffusion model. 
        The standard UNet doesn't include this additional layer, which is used here to map features back to the latent space 
        for the diffusion process.
        """
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.activation = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Output tensor from UNet (Batch, 320, Height/8, Width/8)

        Returns:
            torch.Tensor: Predicted noise (Batch, 4, Height/8, Width/8)
        """
        # (Batch, 320, Height/8, Width/8) -> (Batch, 320, Height/8, Width/8)
        x = self.groupnorm(x)

        # (Batch, 320, Height/8, Width/8) -> (Batch, 320, Height/8, Width/8)
        x = self.activation(x)

        # (Batch, 320, Height/8, Width/8) -> (Batch, 4, Height/8, Width/8)
        x = self.conv(x)

        # (Batch, 4, Height/8, Width/8)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder path: Progressively increase features and reduce spatial dimensions
        self.encoders = nn.ModuleList([
            # Initial projection: map latent space to feature space
            # (Batch, 4, Height/8, Width/8) -> (Batch, 320, Height/8, Width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            # Process and refine features
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            # First downsampling: halve spatial dimensions
            # (Batch, 320, Height/8, Width/8) -> (Batch, 320, Height/16, Width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            # Increase feature channels and apply attention
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),

            # Second downsampling: further reduce spatial dimensions
            # (Batch, 640, Height/16, Width/16) -> (Batch, 640, Height/32, Width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            # Double feature channels and apply attention
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),

            # Final downsampling: reach lowest spatial resolution
            # (Batch, 1280, Height/32, Width/32) -> (Batch, 1280, Height/64, Width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            # Maintain feature channels, focus on abstract representations which can be done with just the residual block
            # Attention is not needed since its at its smallest spatial resolution and it helps save computation resources
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
        ])

        # Bottleneck path: Process the most abstract features at the lowest spatial resolution
        # (Batch, 1280, Height/64, Width/64) -> (Batch, 1280, Height/64, Width/64)
        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),
            UNet_AttentionBlock(8, 160),
            UNet_ResidualBlock(1280, 1280),
        )

        # Decoder path: Progressively decrease features and increase spatial dimensions
        self.decoders = nn.ModuleList([
            # Input is doubled of the output of the bottleneck due to the skip connection from the last layer of encoder path
            # (Batch, 2560, Height/64, Width/64) -> (Batch, 1280, Height/64, Width/64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            # First upsampling: double spatial dimensions and decrease features
            # (Batch, 2560, Height/64, Width/64) -> (Batch, 1280, Height/32, Width/32)
            SwitchSequential(UNet_ResidualBlock(2560, 1280), Upsample(1280)),

            # Process and refine features
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            # Second upsampling: further increase spatial dimensions and decrease features
            # (Batch, 1920, Height/32, Width/32) -> (Batch, 1280, Height/16, Width/16)
            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), Upsample(1280)),

            # Decrease feature channels and apply attention
            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),
            # Third upsampling: reach second highest spatial resolution
            # (Batch, 960, Height/16, Width/16) -> (Batch, 640, Height/8, Width/8)
            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), Upsample(640)),

            # Final processing stages, no upsampling needed here as output shape is (Batch, 320, Height/8, Width/8)
            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
        ])

        # Final output shape: (Batch, 320, Height/8, Width/8)

    # TODO: Document this forward better
    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        # Used to understand the timestep that was used to create the noisy image
        self.time_embedding = TimeEmbedding(320)

        self.unet = UNet()

        # Final layer is used to map features upsampled from UNet back to the original features for the latent space (4)
        self.final = UNet_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: The noisy latent image (variable Z) sampled from the VAE encoder (Batch, 4, Height/8, Width/8).
            context: Embeddings that capture the semantic relationships of the text prompt (Batch, Seq_Len, Dim).
            timestep: The timestep that was used to generate the noisy latent image. (1, 320).
        
        Returns:
            torch.Tensor: Predicted noise by the UNet (Batch, 4, Height/8, Width/8).
        """

        # (1, 320) -> (1, 1280)
        timestep = self.time_embedding(timestep)

        # (Batch, 4, Height/8, Width/8) -> (Batch, 320, Height/8, Width/8)
        output = self.unet(latent, context, timestep)

        # (Batch, 320, Height/8, Width/8) -> (Batch, 4, Height/8, Width/8)
        output = self.final(output)

        # (Batch, 4, Height/8, Width/8)
        return output