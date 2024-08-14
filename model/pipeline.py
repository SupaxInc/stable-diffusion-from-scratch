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
        
        # Load CLIP model and place into device
        clip = models["clip"]
        clip.to_device(device)

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length").input_ids
            # (Batch -> Seq_Len -> Dim)
            cond_tokens = torch.Tensor(cond_tokens, dtype=torch.long, device = device)
            # (Batch -> Seq_Len -> Dim) -> (Batch -> Seq_Len -> Dim)
            cond_context = clip(cond_tokens)
            # (Batch -> Seq_Len -> Dim) -> (Batch -> Seq_Len -> Dim), size of 768

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length").input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

