import torch
import numpy as np
import sys

sys.path.append('../')
from config import DDPMConfig as Config


class DDPMSampler():
    def __init__(self, 
                 generator, 
                 n_training_steps=Config.TRAINING_STEPS,
                 beta_start=Config.BETA_START,
                 beta_end=Config.BETA_END):
        
        self.generator = generator
        self.training_steps = n_training_steps

        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, n_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_dashed = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.timesteps = torch.from_numpy(np.arange(0, n_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_of_timesteps = Config.INFERENCE_STEPS):
        self.inference_steps = num_of_timesteps
        self.skip_ratio = self.training_steps // self.inference_steps
        self.timesteps = torch.from_numpy((np.arange(0, self.inference_steps) * self.skip_ratio).round()[::-1].copy().astype(np.int64))

    def add_noise(self, latents: torch.FloatTensor, timestep: torch.IntTensor):
        alphas_dashed = self.alphas_dashed.to(device=latents.device, dtype=latents.dtype)
        timestep = timestep.to(latents.device)

        sqrt_alphas_dashed = alphas_dashed[timestep] ** 0.5
        sqrt_alphas_dashed = sqrt_alphas_dashed.flatten()
        while sqrt_alphas_dashed.dim() < latents.dim():
            sqrt_alphas_dashed = sqrt_alphas_dashed.unsqueeze(-1)

        sqrt_one_minus_alpha_dashed = (1 - alphas_dashed[timestep]) ** 0.5
        sqrt_one_minus_alpha_dashed = sqrt_one_minus_alpha_dashed.flatten()
        while sqrt_one_minus_alpha_dashed.dim() < sqrt_one_minus_alpha_dashed.dim():
            sqrt_one_minus_alpha_dashed = sqrt_one_minus_alpha_dashed.unsqueeze(-1)

        noise = torch.randn(latents.shape, generator=latents.generator, dtype=latents.dtype, device=latents.device)
        # CHECK IF IT'S WORKING PROPERLY !!!
        noised_image = sqrt_alphas_dashed * latents + noise * sqrt_one_minus_alpha_dashed

        return noised_image

    def step(self, timestep, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = timestep - self.skip_ratio

        alpha_dashed_t = self.alphas_dashed[t]
        alpha_dashed_prev_t = self.alphas_dashed[prev_t] if prev_t >= 0 else self.one

        beta_dashed_t = 1.0 - alpha_dashed_t
        beta_dashed_prev_t = 1.0 - alpha_dashed_prev_t

        current_alpha_t = alpha_dashed_t/alpha_dashed_prev_t
        current_beta_t = 1.0 - current_alpha_t

        original_sample_prediction = (latents - beta_dashed_t ** (0.5) * model_output)/ alpha_dashed_t ** (0.5)
        mean_coeff_x_o = (alpha_dashed_prev_t ** (0.5) * current_beta_t) / beta_dashed_t
        mean_coeff_x_t = (current_alpha_t ** (0.5) * beta_dashed_prev_t) / beta_dashed_t

        mean = mean_coeff_x_o*original_sample_prediction + mean_coeff_x_t*latents

        # variance
        variance = 0
        if t > 0:
            noise = torch.randn(model_output.shape,
                                generator=self.generator,
                                dtype=model_output.dtype,
                                device=model_output.device)
            variance = (beta_dashed_prev_t / beta_dashed_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            variance = variance ** 0.5
            variance = variance * noise

        return mean + variance

    def set_strength(self, strength: float = 1.0):
        self.start_step = self.inference_steps - int(self.inference_steps*strength)
        self.timesteps = self.timesteps[self.start_step:]
