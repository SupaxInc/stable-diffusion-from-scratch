import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.ddpm import DDPMSampler
from tqdm import tqdm

# Stable Diffusion can only accept 512x512
WIDTH = 512
HEIGHT = 512

LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# TODO: Refactor this to have separate functions for text-to-image, image-to-image, inpaiting
def generate(
        prompt: str, 
        uncond_prompt: str, 
        input_image: str=None, 
        strength=0.8, 
        do_cfg=True, 
        cfg_scale=7.5, 
        sampler_name="ddpm", 
        n_inference_steps=50, 
        models={}, 
        seed=None,
        device=None, 
        idle_device=None, 
        tokenizer=None,
):
    """
    Generate an image using the Stable Diffusion pipeline.

    Args:
        prompt (str): The text prompt to guide the image generation.
        uncond_prompt (str): The unconditional prompt used for classifier-free guidance.
        input_image (str, optional): Path to an input image for image-to-image generation. Defaults to None.
        strength (float, optional): Determines how much to transform the input image. Only used if input_image is provided. Defaults to 0.8.
        do_cfg (bool, optional): Whether to use classifier-free guidance. Defaults to True.
        cfg_scale (float, optional): The scale for classifier-free guidance. Higher values result in stronger adherence to the prompt. Defaults to 7.5.
        sampler_name (str, optional): The name of the sampler to use. Currently only supports "ddpm". Defaults to "ddpm".
        n_inference_steps (int, optional): The number of denoising steps. Defaults to 50.
        models (dict): A dictionary containing the required models: "clip", "encoder", "diffusion", and "decoder".
        seed (int, optional): Seed for the random number generator. If None, a random seed will be used. Defaults to None.
        device (torch.device, optional): The device to run the generation on. Defaults to None.
        idle_device (torch.device, optional): The device to move models to when not in use. Defaults to None.
        tokenizer: The tokenizer to use for encoding the prompts.

    Returns:
        numpy.ndarray: The generated image as a numpy array with shape (HEIGHT, WIDTH, 3) and dtype uint8 so that it can be visualized later.
    """
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1!")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Generate a random number or use a known seed
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Prepare to do two inferences for Classifier-free guidance, one for conditioned output and another for unconditioned output

            # Convert the prompt into tokens using the tokenizer, if it's too short fill it with paddings
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # Convert input ids (tokens) to a tensor: (1, 77), (Batch, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # Convert the tokens to embeddings: (1, 77) -> (1, 77, 768), where 768 is the CLIP embedding dimension
            cond_context = clip(cond_tokens)

            # Do the same for unconditioned output
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            # (1, 77)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (1, 77) -> (1, 77, 768), (Batch, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)

            # Concatenate the two outputs so it's prepared to be used as an input to U-Net
            # (1, 77, 768) + (1, 77, 768) = (2, 77, 768), (Batch, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Without classifier-free guidance
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (1, 77)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77) -> (1, 77, 768),(Batch, Seq_Len, Dim)
            context = clip(tokens)

        print(f"CLIP encoding shape: {context.shape}")
        print(f"CLIP encoding stats: mean={context.mean().item():.4f}, std={context.std().item():.4f}")
        
        # Can offload CLIP model to CPU or whatever device when not being used
        to_idle(clip)

        # Define the sampler and set the number of inference steps
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        # Corresponds to the output tensor shape of the VAE encoder
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image:
            # Begin the image-to-image pipeline

            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (HEIGHT, WIDTH, 3)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # Rescale pixel values from [0, 255] to [-1, 1] for the VAE which will be used as an input to U-Net
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (HEIGHT, WIDTH, 3) -> (1, HEIGHT, WIDTH, 3)
            input_image_tensor = input_image_tensor.unsqueeze(0)

            # VAE Encoder requires the Channel dimension to be before Height and Width
            # (1, HEIGHT, WIDTH, 3) -> (1, 3, HEIGHT, WIDTH)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Generate noise for the encoder (allows deterministic results with a seed)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # Run the image through the VAE encoder to get the latent representation (Z)
            # (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH), output of the VAE encoder
            latents = encoder(input_image_tensor, encoder_noise)
            
            # Set the strength which shifts the timesteps
                # Higher strength = Start with more noise
                # Lower strength = Start with less noise
            sampler.set_strength(strength=strength)
            # Add noise to the latents based on new timesteps, less noise preverse more of the original image
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # Begin text-to-image pipeline

            # Start with random noise in the latent space: N(0, I)
            # (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        print(f"Initial latents shape: {latents.shape}")
        print(f"Initial latents mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        # Iteratively denoise the latents for every timestep (based on # of inference steps)
        for i, timestep in enumerate(timesteps):
            # Create a time embedding for the current timestep
            # (1,) -> (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
            model_input = latents

            if do_cfg:
                # Duplicate the input so there are two latents for conditional and unconditional outputs
                # (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH) -> (2, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
                model_input = model_input.repeat(2, 1, 1, 1)
            
            # Predict noise using the UNet
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                # Apply classifier-free guidance: z_guided = z_uncond + cfg_scale * (z_cond - z_uncond)
                model_output = output_uncond + cfg_scale * (output_cond - output_uncond)
            
            # Update latents by removing the predicted noise
            latents = sampler.step(timestep, latents, model_output)
            
            print(f"Step {i}, timestep: {timestep}")
            print(f"UNet input stats: mean={model_input.mean().item():.4f}, std={model_input.std().item():.4f}")
            print(f"UNet output stats: mean={model_output.mean().item():.4f}, std={model_output.std().item():.4f}")
            print(f"Updated latents stats: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
        
        to_idle(diffusion)
        print(f"Final latents stats: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
        decoder = models["decoder"]
        decoder.to(device)
        # Decode the latents to get the final image
        images = decoder(latents)

        to_idle(decoder)
        
        # Rescale pixel values from [-1, 1] to [0, 255]
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (1, 3, HEIGHT, WIDTH) -> (1, HEIGHT, WIDTH, 3)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy() # Convert to numpy array so it can be visualized later

        print(f"Decoded image stats: min={images.min().item():.4f}, max={images.max().item():.4f}, mean={images.mean().item():.4f}")

        # Return the first (and only) image in the batch
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    """
    Rescale the tensor `x` from one range to another.

    This function adjusts the values in tensor `x` from an original range (`old_range`)
    to a new range (`new_range`). Optionally, values can be clamped to the new range
    to ensure no values fall outside it. This is particularly useful in image processing
    where pixel values need to be adjusted between different scales, e.g., [-1, 1] to [0, 255].

    Args:
        x (torch.Tensor): The input tensor to rescale.
        old_range (tuple): A tuple (old_min, old_max) representing the minimum and
                           maximum values of the current range of `x`.
        new_range (tuple): A tuple (new_min, new_max) representing the target minimum
                           and maximum values for `x` after rescaling.
        clamp (bool, optional): If True, values will be clamped to the range specified
                                by `new_range`. Default is False.

    Returns:
        torch.Tensor: The rescaled tensor with the same dimensions as the input `x`.
    """
    # Subtract the old minimum from `x` to shift values starting from zero
    old_min, old_max = old_range
    new_min, new_max = new_range
    # Shifts all values in tensor x by -old_min (E.g. [10, 20, 30] = [0, 10, 30])
    x -= old_min
    # Scale the shifted values in tensor x to the new range
    x *= (new_max - new_min) / (old_max - old_min)
    # Add the new minimum to shift the scaled values to the desired range
    x += new_min

    # Optionally clamp values to ensure they remain within the new range
    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x

def get_time_embedding(timestep):
    """
    Generate a time embedding feature vector for a given timestep in the diffusion process.

    Creates a time-dependent signal used in Stable Diffusion to control
    the denoising process at different stages. It uses sinusoidal positional encoding
    to generate a unique embedding for each timestep.

    The formula here is taken from the original Transformer architecture so its similar to how Transformers
    encode position information for tokens in a sequence.

    Args:
        timestep (int): The current timestep in the diffusion process.

    Returns:
        torch.Tensor: A tensor of shape (1, 320) containing the time embeddings.
    """
    # Calculate frequencies for positional encoding
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    
    # Scale timestep by frequencies
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    
    # Apply sine and cosine functions and concatenate
    # Shape: (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)