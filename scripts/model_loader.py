from model.clip import CLIP
from model.encoder import VAE_Encoder
from model.decoder import VAE_Decoder
from model.diffusion import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    
    # Load the state of the pre trained model and strictly map the names
        # Ensure that the names are correctly mapped in model_converter.py

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True) #

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)
    
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict=["clip"], strict=True)

    return {
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
        "clip": clip,
    }