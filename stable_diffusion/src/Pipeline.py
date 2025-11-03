import torch
import numpy as np
from tqdm import tqdm

from config import CLIPConfig
from config import DiffusionConfig
from DDPM import DDPMSampler


class Pipeline():
    def __init__(self):
        self.IMAGE_HEIGHT = 512
        self.IMAGE_WIDTH = 512
        self.LATENT_HEIGHT = self.IMAGE_HEIGHT//8
        self.LATENT_WIDTH = self.IMAGE_WIDTH//8

    def rescale(self, x: torch.Tensor, old_range, new_range, clamp: bool = False) -> torch.Tensor:
        old_low, old_up = old_range
        new_low, new_up = new_range

        x -= old_low
        x *= (new_up-new_low)/(old_up-old_low)
        x += new_low
        
        if clamp:
            x = x.clamp(new_low, new_up)

        return x

    def get_time_embeddings(self, timestep: int):
        freqs = torch.pow(10_000, (-torch.arange(0, DiffusionConfig.D_TIME//2, dtype=torch.float32)/(DiffusionConfig.D_TIME//2))) # (160,)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None] # (1, 160)
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # (1, 320)

        return x


    def generate(self,
                 prompt,
                 uncond_prompt,
                 input_image = None,
                 strength = 0.8,
                 cfg = True,
                 scale_cfg = 7.5,
                 sampler_name = "ddpm",
                 inference_steps = 50,
                 models = {},
                 seed = None,
                 device = None,
                 idle_device = None,
                 tokenizer = None
                 ):
        with torch.no_grad():
            assert 0 < strength <= 1

            if idle_device:
                to_idle = lambda x: x.to(idle_device)
            else:
                to_idle = lambda x: x

            generator = torch.Generator(device)
            if seed:
                generator.manual_seed(seed)
            else: generator.seed()

            clip = models['clip'].to(device)

            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=CLIPConfig.SEQ_LEN).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_tokens = clip(cond_tokens) # (1, seq_len, D)
            context = None
            if cfg:
                uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=CLIPConfig.SEQ_LEN).input_ids
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
                uncond_tokens = clip(uncond_tokens) # (1, seq_len, D)

                context = torch.cat([cond_tokens, uncond_tokens], dim=0) # (2, seq_len, D)
            else:
                context = cond_tokens # (1, seq_len, D)

            to_idle(clip)

            assert sampler_name == 'ddpm'
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(inference_steps)

            if input_image:
                encoder = models['encoder'].to(device)
                input_img_tensor = input_image.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
                input_img_tensor = np.array(input_img_tensor)
                input_img_tensor = torch.tensor(input_img_tensor, dtype=torch.float32, device=device)
                input_img_tensor = self.rescale(input_img_tensor, (0, 255), (-1, 1))
                input_img_tensor = input_img_tensor.unsqueeze(0) # (1, H, W, Channels)
                input_img_tensor = torch.permute(input_img_tensor, (0, 3, 1, 2)) # (1, Channels, H, W)

                encoder_noise = torch.randn((1, 4, self.LATENT_HEIGHT, self.LATENT_WIDTH), generator=generator, device=device)
                latents = encoder(input_img_tensor, encoder_noise)

                sampler.set_strength(strength)
                latents = sampler.add_noise(latents, sampler.timesteps[0])
                to_idle(encoder)
            else:
                latents = torch.randn((1, 4, self.LATENT_HEIGHT, self.LATENT_WIDTH), generator=generator, device=device)

            diffusion = models['diffusion'].to(device)
            timesteps = tqdm(sampler.timesteps)
            for i,timestep in enumerate(timesteps): 
                time_embd = self.get_time_embeddings(timestep).to(device) # (1, 320)

                m_input = latents
                if cfg:
                    m_input = m_input.repeat((2, 1, 1, 1)) # one for cond_in, another for uncond_in

                m_output = diffusion(m_input, context, time_embd) # (B, 4, H/8, W/8)

                if cfg:
                    cond_out, uncond_out = torch.chunk(m_output, 2, dim=0)
                    m_output = scale_cfg*(cond_out-uncond_out) + uncond_out # (1, 4, H/8, H/8)

                latents = sampler.step(timestep, latents, m_output)

            to_idle(diffusion)

            decoder = models['decoder'].to(device)
            out_img = decoder(latents).to(device) # (1, 3, H, W)
            to_idle(decoder)

            out_img = self.rescale(out_img, (-1, 1), (0, 255), clamp=True)
            out_img = out_img.permute(0, 2, 3, 1) # (1, H, W, 3)
            out_img = out_img.to('cpu', torch.uint8).numpy()

            return out_img[0]

