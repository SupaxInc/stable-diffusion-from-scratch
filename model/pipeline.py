import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ddpm import DDPMSampler

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
        clip.to_device(device)

        if do_cfg:
            # Prepare to do two inferences for Classifier-free guidance, one for conditioned output and another for unconditioned output

            # Convert the prompt into tokens using the tokenizer, if its too short fill it with paddings
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # Convert input ids (tokens) to a tensor: (Batch, Seq_Len)
            cond_tokens = torch.Tensor(cond_tokens, dtype=torch.long, device=device)
            # Convert the tokens to embeddings: (Batch, Seq_Len) -> (Batch, Seq_Len, Dim), dim is size of 768 (from CLIP embed dims)
            cond_context = clip(cond_tokens)

            # Do the same for unconditioned output
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length").input_ids
            # (Batch, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch, Seq_Len) -> (Batch, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)

            # Concatenante the two outputs so its prepared to be used as an input to U-Net
            # (Batch, Seq_Len, Dim) + (Batch, Seq_Len, Dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Classifier guidance 
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch, Seq_Len)
            tokens = torch.Tensor(tokens, dtype=torch.long, device=device)
            # (Batch, Seq_Len) -> (Batch, Seq_Len, Dim) = (1, 77, 768)
            context = clip(tokens)
        
        # Can offload CLIP model to CPU or whatever device when not being used
        to_idle(clip)

        # Define the amount denoisification steps for the sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler: {sampler}")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        if input_image:
            # Begin the image-to-image architecture 

            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # Each pixel is from 0 to 255 which is incorrect for U-Net inputs, U-Net needs pixels to be clamped from -1 to 1 instead
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (Height, Width, Channel) -> # (Batch, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)

            # VAE Encoder requires the Height and Width to be the last 2 index of the tensor shape
            # (Batch, Height, Width, Channel) -> # (Batch, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Allows us to make the noise deterministic if we have a seed
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # Run the image through the VAE encoder, create the latent (Z)
            latents = encoder(input_image_tensor, encoder_noise)
            # Use the strength parameter to generate the noisy image
            # A noisier image will give more freedom for creativity, while a less noisy image will look more like the original image
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # Begin text-to-image architecture

            # Begin with a random noise N(0, I)
            latents = torch.randn(latents_shape, generator=generator, device=device)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        # Continuously denoise the image for every timestep (based on # of inference steps)
        for i, timestep in enumerate(timesteps):
            # Turn the scalar timestep into a vector: (1, 320) -> (1, 1280)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch, 4, Latent_Height, Latent_Width)
            model_input = latents

            if do_cfg:
                # Create two copies of the latent so it can be used for condition and unconditioned prompts
                # (Batch, 4, Latent_Height, Latent_Width) -> (2 * Batch, 4, Latent_Height, Latent_Width)
                model_input = model_input.repeat(2, 1, 1, 1)
            
            # Predicted noise by the UNet
            model_output  = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                # Classifier-free guidance formula: zguided = zuncond + w * (zcond - zuncond)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # Remove noise predicted by the UNet
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # Run the images to the decoder to rescale the image back
        images = decoder(latents)

        to_idle(decoder)
        
        # Begin rescaling the image back to 0-255 for RGB
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch, Channel, Height, Width) -> (Batch, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)