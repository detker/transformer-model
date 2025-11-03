import sys
import torch
import torch.nn as nn

sys.path.append('../')
from config import DiffusionConfig as Config
from diffusion.time_embeddings import TimeEmbeddings
from diffusion.unet import UNet
from diffusion.unet_output_layer import UNetOutputLayer


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embeddings = TimeEmbeddings(Config.D_TIME)
        self.unet = UNet()
        self.unet_output_layer = UNetOutputLayer(Config.D_UNET_DECODER_OUT, Config.D_DIFFUSION_IN_OUT)

    def forward(self, latents: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latents: (B, 4, H/8, W/8), context: (B, seq_len, D), time: (1, D_TIME=320)

        # (1, 1280)
        time_embd = self.time_embeddings(time)

        # (B, 320, H/8, W/8)
        out = self.unet(latents, context, time_embd)

        # (B, 4, H/8, H/8)
        out = self.unet_output_layer(out)

        return out
