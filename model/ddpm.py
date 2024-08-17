import torch
import numpy as np

class DDPMSampler:
    def __init__(
            self, 
            generator: torch.Generator, 
            num_training_steps: int = 1000,
            num_inference_steps: int = 50, 
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
            # alpha_0 * alpha_1 * alpha_2 if t = 2
        # This corresponds to ᾱ_t (alpha bar) in the DDPM paper
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        
        # Constant used in various calculations
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps
        self.step_ratio = self.num_training_steps // self.num_inference_steps
        
        # Create an array of timesteps in reverse order
        # This is used to iterate through the diffusion process from T to 0
            # tensor([999, 998, 997, ..., 2, 1, 0]) if T is 1000 
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        
    def set_inference_steps(self, num_inference_steps: int = 50) -> torch.Tensor:
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
        self.step_ratio = self.num_training_steps // self.num_inference_steps
        
        # Create an array of timesteps for inference, evenly spaced across the original training steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)

        # Convert the numpy array to a PyTorch tensor
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.step_ratio
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Computes β̄_t (beta bar t), which represents the variance in the reverse process.
        
        This variance is derived from equations 6 and 7 in the DDPM paper:
        Eq. 6: q(x_t | x_{t-1}) = N(x_t; √(1 - β_t) * x_{t-1}, β_t * I)
        Eq. 7: q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t(x_t, x_0), β̃_t * I)
        
        Where β̄_t is used to determine the variance of the Gaussian distribution
        for the reverse diffusion process.
        """
        prev_t = self._get_previous_timestep(timestep)

        # ᾱ_t (alpha bar t)
        alpha_cumprod_t = self.alpha_cumprod[timestep]
        # ᾱ_{t-1} (alpha bar t-1)
        alpha_cumprod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        # β_t (beta t)
        current_beta_t = 1 - alpha_cumprod_t / alpha_cumprod_t_prev
        
        # Compute β̄_t (beta bar t) using the formula derived from equations 6 and 7
        # β̄_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t * β_t)
        beta_bar_t = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t * current_beta_t)
        # Clamp to prevent numerical instability
        beta_bar_t = torch.clamp(beta_bar_t, min=1e-20)

        return beta_bar_t
    
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.FloatTensor:
        """
        Reverse process of the DDPM sampler. Removes the noise from the noisified image. x_t -> x_t-1

        Args:
            timestep (int): Current timestep t in the diffusion process. 
                            Variable: t
            latents (torch.Tensor): The current noisy latent images. 
                            Variable: x_t
            model_output (torch.Tensor): The predicted noise at current timestep by the model. 
                            Variable: ε_θ(x_t, t)

        Returns:
            torch.FloatTensor: The denoised latent image x_t-1.

        This method implements the reverse process using the following equations from the DDPM paper:
        1. Equation (15): x_0 = (x_t - √(1 - α̅_t) * ε_θ(x_t, t)) / √(α̅_t)
        2. Equation (7): μ_θ(x_t, t) = √(α̅_t-1)β_t / (1 - α̅_t) * x_0 + √(α_t)(1 - α̅_t-1) / (1 - α̅_t) * x_t
        3. Equation (11): p_θ(x_t-1 | x_t) = N(x_t-1; μ_θ(x_t, t), Σ_θ(x_t, t))
        4. Equation derived from 11 to sample previous noisy image x_t-1: xt-1 = μθ(xt, t) + σt * z
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)  # E.g., if t is 980, this will return 960 for 50 inference steps
        
        # α̅_t and α̅_t-1 from the cumulative product of (1 - β_t)
        # These will be used in equations (15) and (7)
        alpha_cumprod_t = self.alpha_cumprod[timestep]
        alpha_cumprod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one

        # β̅_t and β̅_t-1
        # These will be used in equation (7)
        beta_cumprod_t = 1 - alpha_cumprod_t
        beta_cumprod_t_prev = 1 - alpha_cumprod_t_prev

        # α_t = α̅_t / α̅_t-1
        # This will be used in equation (7)
        current_alpha_t = alpha_cumprod_t / alpha_cumprod_t_prev
        # β_t = 1 - α_t
        # This will be used in equation (7)
        current_beta_t = 1 - current_alpha_t

        # Compute x_0 using equation (15)
        # x_0 = (x_t - √(1 - α̅_t) * ε_θ(x_t, t)) / √(α̅_t)
        pred_original_sample = (latents - beta_cumprod_t ** 0.5 * model_output) / alpha_cumprod_t ** 0.5

        # Preparing for equation (7)
        # Compute coefficients for pred_original_sample (x_0) and current sample (x_t)
        # √(α̅_t-1)β_t / (1 - α̅_t)
        pred_original_sample_coeff = (alpha_cumprod_t_prev ** 0.5 * current_beta_t) / beta_cumprod_t
        # √(α_t)(1 - α̅_t-1) / (1 - α̅_t)
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_cumprod_t_prev) / beta_cumprod_t

        # Compute the predicted previous sample mean using equation (7)
        # μ_θ(x_t, t) = √(α̅_t-1)β_t / (1 - α̅_t) * x_0 + √(α_t)(1 - α̅_t-1) / (1 - α̅_t) * x_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # Compute σ_t (sigma_t) for the equation: x_t-1 = μ_θ(x_t, t) + σ_t * z
        # This is derived from equation (11) in the DDPM paper: p_θ(x_t-1 | x_t) = N(x_t-1; μ_θ(x_t, t), Σ_θ(x_t, t))
        sigma_t = 0

        # From the sampling algorithm, no need to generate random noise if we are already back to original x_0
            # This means we have finished denoising already
        if t > 0:
            device = model_output.device

            # Generate random noise: z ~ N(0, I)
            z = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)

            # Compute σ_t (sigma_t), which is the square root of the variance (Σ_θ(x_t, t))
                # σ_t = √(β_t * (1 - α̅_t-1) / (1 - α̅_t))
                # Square root of variance to get σ_t (standard deviation)
                # σ_t * z represents the random noise scaled by the standard deviation
            sigma_t = (self._get_variance(t) ** 0.5) * z

        # Compute the equation: x_t-1 = μ_θ(x_t, t) + σ_t * z
        pred_prev_sample = pred_prev_sample + sigma_t

        return pred_prev_sample

    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor) -> torch.FloatTensor:
        """
        Forward process of the DDPM sampler. Adds noise to the original samples (input images). x_0 -> x_t

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

        # Calculate mean: sqrt(α_t)
        sqrt_alpha_cumprod = alpha_cumprod[timestep] ** 0.5 # Exponent 0.5 is same as sqrt
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

        # Generate random noise with the same shape as original images
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)

        # Begin sampling from the distribution (generating random noise), according from equation (4, forward process) of the DDPM paper
            # Similar to formula to transform normal variable to desired distribution: X = mean + stdev * z
        noisy_samples = (sqrt_alpha_cumprod * original_samples) + (sqrt_one_minus_alpha_cumprod * noise)

        return noisy_samples
