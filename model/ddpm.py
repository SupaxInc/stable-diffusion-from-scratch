import torch
import numpy as np

class DDPMSampler:
    def __init__(
            self, 
            generator: torch.Generator, 
            num_training_steps: int = 1000, 
            beta_start: float = 0.00085, 
            beta_end: float = 0.0120
    ):
        """
        Args:
            generator (torch.Generator): Random number generator for reproducibility.
            num_training_steps (int): Total number of steps in the forward diffusion process.
            beta_start (float): Starting value for the noise schedule.
            beta_end (float): Ending value for the noise schedule.

        The DDPM sampler is responsible for the forward and reverse diffusion processes.
        It sets up the noise schedule and calculates important parameters used in the diffusion process.
        """
        # Create noise schedule (betas) using a quadratic function
        # This corresponds to β_t in the DDPM paper
            # torch.linspace creates a 1D tensor of num_training_steps evenly spaced values from beta_start**0.5 to beta_end**0.5
            # We use **0.5 (square root) and later **2 (square) to create a quadratic noise schedule
            # This results in a smoother transition of noise levels compared to a linear schedule
            # The quadratic schedule allows for more controlled noise addition in the early steps and faster noise increase in later steps
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        
        # Calculate alphas: α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # Calculate cumulative product of alphas [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...., etc]
        # This corresponds to ᾱ_t (alpha bar) in the DDPM paper
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        
        # Constant used in various calculations
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        
        # Create an array of timesteps in reverse order
        # This is used to iterate through the diffusion process from T to 0
            # tensor([999, 998, 997, ..., 2, 1, 0]) if T is 1000 
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        
    def set_inference_steps(self, num_inference_steps: int=50) -> torch.Tensor:
        """
        Calculates a subset of timesteps from the original training steps,
        allowing the model to generate images more quickly while maintaining quality.

        Args:
            num_inference_steps (int): The number of inference steps to use. Default is 50.

        The difference between inference steps and training steps:
        - Training steps (num_training_steps): The total number of steps used in the forward
          diffusion process during training, typically a large number like 1000.
        - Inference steps (num_inference_steps): The number of steps used in the reverse
          diffusion process during generation, typically fewer than training steps for
          efficiency, e.g., 50.
        """
        self.num_inference_steps = num_inference_steps

        # Need to customize the amount of steps decreased depending on number of inference steps set
            # 999, 998, 997, 996, ... = 1000 steps -> 1000/1000 = 1 step decrease
            # 999, 999-20, 999-40, ... = 50 steps -> 1000/50 = 20 steps decrease
        step_ratio = self.num_training_steps // self.num_inference_steps
        
        # Create an array of timesteps for inference, evenly spaced across the original training steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

        # Convert the numpy array to a PyTorch tensor
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor) -> torch.FloatTensor:
        """
        Adds noise to the original samples (input images).

        Args:
            original_samples (torch.FloatTensor): The original, clean samples to which noise will be added  (Batch, Channels, Height, Width).
            timestep (torch.IntTensor): The current timestep in the diffusion process (Batch, ).

        Returns:
            torch.FloatTensor: The noisy samples after adding noise.
                Shape: Same as original_samples

        This method follows Equation (4) forward process from the DDPM paper, 
        which describes transitioning from the original image (x0) to any noisified image (xt) in one step:
        
        q(x_t | x_0) = N(x_t; sqrt(α_t)x_0, (1 - α_t)I)
        """
        # Move alpha_cumprod to the same device and dtype as original_samples
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        # Ensure timestep is on the correct device
        timestep = timestep.to(original_samples.device)

        # Calculate sqrt(α_t)
        sqrt_alpha_cumprod = alpha_cumprod[timestep] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
        # Expand dimensions until it matches original_samples
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)

        # Calculates standard deviation (not variance): sqrt(1 - α_t)
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod[timestep]) ** 0.5
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()
        # Expand dimensions until it matches original_samples
        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        # Begin sampling from the distribution (generating random noise), according from equation (4, forward process) of the DDPM paper
            # Similar to formula to transform normal variable to desired distribution: X = mean + stdev * z
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)

        # Combine the original samples with noise according to the DDPM equation
        noisy_samples = (sqrt_alpha_cumprod * original_samples) + (sqrt_one_minus_alpha_cumprod * noise)

        return noisy_samples
