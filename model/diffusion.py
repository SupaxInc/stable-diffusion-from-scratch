import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        """
        Convert a scalar timestep that is used for the generated noisy latent into a feature vector (time embedding).
        """
        super().__init__()

        self.activation = nn.SiLU()  # SiLU activation (also known as Swish)

        # Feed forward network
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, 4 * embed_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        In diffusion models, each noise level (timestep) needs to be represented
        as a vector to provide temporal information to the model. This embedding
        allows the model to understand and differentiate between different stages
        of the diffusion process.

        Args:
            t: Input timestep tensor (1, 320).
        
        Returns:
            torch.Tensor: Time embedding (Batch, 4*embed_dim)
        """
        # Expand dimensionality of the time step
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
        Residual blocks allows us to preserve information and ensure stable gradients.

        Args:
            in_channels: Input channels of tensor that is being used.
            out_channels: Output channels to transform the tensor to.
            time_embedding_features: The features of the timestep when it has been converted to a feature vector (scale factor of 4).
        """
        super().__init__()
        self.group_norm_feature = nn.GroupNorm(32, in_channels)
        self.activation = nn.SiLU()
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(time_embedding_features, out_channels)

        self.group_norm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            # If the number of input channels is equal to the number of output channels, use an identity layer for the residual connection.
            # The identity layer passes the input directly to the output without any changes.
            self.residual_layer = nn.Identity
        else:
            # If its different, use a 1x1 convolution to match the input channel as output channel for the residual connection.
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, z: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        Incorporate the time information with the latent to help model understand the noise level at each step of the diffusion process, 
        allowing it to gradually denoise the image.

        Args:
            z: Latent representation (z) of noise generated for current timestep from encoder (Batch, 4, Height/8, Width/8).
            time_embedding: Timestep feature vector for current noise level (Batch, Dim * 4) -> (1, 1280).
        """

        residue = z

        z = self.group_norm_feature(z)
        z = self.activation(z)
        z = self.conv_feature(z)

        time_embedding = self.activation(time_embedding)
        time_embedding = self.linear_time(time_embedding)

        # Merging the latent with the time embedding
            # Adding height and width dimension to time embedding feature vector
        merged = z + time_embedding.unsqueeze(-1).unsqueeze(-1)
        merged = self.group_norm_merged(merged)
        merged = self.activation(merged)
        merged = self.conv_merged(merged)
    
        return merged + self.residual_layer(residue)
    
class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        A custom Sequential module to allow flexibility of handling different types of layers in a U-Net architecture.
        The UNet will have multiple inputs to guide the diffusion model to the correct image. We need a way to process these
        different inputs for every layer.

        Args:
            x: Latent representation (z) at the current noise level (Batch, Channels/Features, Height, Width).
            context: Text embedding from the CLIP text encoder (Batch, Seq_Len, Dim).
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
                # Residual blocks incorporate time information
                # Helps model understand the noise level at each step of the diffusion process, allowing it to gradually denoise the image
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
        self.group_norm = nn.GroupNorm(32, in_channels)
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
        x = self.group_norm(x)

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
        """

        # (1, 320) -> (1, 1280)
        timestep = self.time_embedding(timestep)

        # (Batch, 4, Height/8, Width/8) -> (Batch, 320, Height/8, Width/8)
        output = self.unet(latent, context, timestep)

        # (Batch, 320, Height/8, Width/8) -> (Batch, 4, Height/8, Width/8)
        output = self.final(output)

        # (Batch, 4, Height/8, Width/8)
        return output