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

def generate(prompt: str, uncond_prompt: str, input_image: str=None, 
             strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", 
             n_inference_steps=50, models={}, seed=None,
             device=None, idle_device=None, tokenizer=None,
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
            # Prepare to do two inferences for classifier free guidance, one for conditioned output and another for unconditioned output

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
            tokens = torch.Tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        
        # Can offload to CPU or whatever device in the mean time
        to_idle(clip)

        
