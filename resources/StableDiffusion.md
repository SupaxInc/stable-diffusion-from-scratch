# How Stable Diffusion Works
## What is it?
It is a generative model that learns the probability distribution of a data set (images in this case) and can generate entirely new images by sampling from this distribution.

# Process Overview of Stable Diffusion
1. Autoencoder:
- Encoding: The initial image is encoded into a latent space using an autoencoder.
- Components: This autoencoder consists of an encoder that compresses the image into a latent representation and a decoder that can reconstruct the image from this latent representation.

2. Latent Space Representation:
- The image, now represented in a compressed latent space, can be more efficiently processed for tasks like denoising.
- This latent representation retains essential features and context of the image but in a lower-dimensional space, making subsequent processing computationally more efficient.

3. Denoising U-Net with Attention Mechanisms:
- The latent representation is fed into a U-Net architecture, enhanced with self-attention and cross-attention blocks, which is used for the denoising process in the diffusion model.
- Self-Attention Blocks: Capture long-range dependencies and contextual information within the same resolution level.
- Cross-Attention Blocks: Enable interaction between different resolution levels and between the latent image features and textual features from the CLIP model.

4. Diffusion Process:
- The diffusion process involves adding noise to the latent representation (forward process) and then iteratively denoising it (reverse process).
- **DDPM** (Denoising Diffusion Probabilistic Models): The forward process gradually adds Gaussian noise, and the reverse process probabilistically denoises the image using a U-Net.
- **DDIM** (Denoising Diffusion Implicit Models): Similar to DDPM but uses a non-Markovian and potentially deterministic approach for the reverse process, allowing for more efficient and consistent denoising with fewer steps.

5. Guidance with CLIP:
- In guided diffusion models, mechanisms like CLIP are used to ensure the generated image aligns with a text prompt.
- Text Encoding: The text prompt is encoded into a latent vector using the CLIP text encoder.
- Image Encoding: Intermediate images are encoded into latent vectors using the CLIP image encoder.
- Comparison and Adjustment: The CLIP latent vectors of the image and text prompt are compared, guiding the U-Net’s denoising steps to ensure the final output accurately reflects the text description.

6. Reconstruction:
- After the denoising process, the refined latent representation is passed through the decoder part of the autoencoder.
- The decoder reconstructs the final image from the denoised latent representation.

## Detailed Breakdown
1. Encoding with Autoencoder:
- Input Image: The process starts with an input image.
- Encoder: The encoder compresses this image into a latent representation, capturing essential features and context.

2. Diffusion Process in Latent Space:
- Noise Addition: Noise is added to the latent representation.
- U-Net Denoising with Attention: The noisy latent representation is iteratively denoised by the U-Net.
  - Self-Attention Blocks: Capture and enhance features by considering dependencies within the same resolution level.
  - Cross-Attention Blocks: Facilitate the integration of textual features (from CLIP) with image features, and enable interactions between different resolutions.

3. Guidance (if applicable):
- CLIP Encoding: Both the intermediate latent representation and the text prompt are encoded using CLIP.
- Comparison and Adjustment: The CLIP latent vectors are compared to guide the U-Net’s denoising steps, ensuring alignment with the text prompt.

4. Reconstruction with Decoder:
- Final Latent Representation: After iterative denoising, the final latent representation is obtained.
- Decoder: This representation is decoded back into the image space to produce the final output.

## Simplified Visualization
```
Input Image
   |
[Autoencoder Encoder]
   |
Latent Representation (compressed)
   |
[Diffusion Process in Latent Space with Self-Attention and Cross-Attention]
   |   \
Noise   Denoising (U-Net with Attention)
   |     /
Intermediate Latent Representations (iteratively refined)
   |
Final Latent Representation
   |
[Autoencoder Decoder]
   |
Final Image
```