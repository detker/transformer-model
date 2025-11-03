import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append('../')
from config import VAEEncoderConfig as Config
from VAE.residual_block import ResidualBlock
from attention import AttentionBlock


class VAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # in: (B, 3, H, W) -> out: (B, 8, H/8, W/8)
            nn.Conv2d(Config.IMG_CH, Config.IN_CH, kernel_size=3, padding=1),
            ResidualBlock(Config.IN_CH, Config.IN_CH),
            ResidualBlock(Config.IN_CH, Config.IN_CH),

            nn.Conv2d(Config.IN_CH, Config.IN_CH, kernel_size=3, stride=2, padding=0),
            ResidualBlock(Config.IN_CH, 2*Config.IN_CH),
            ResidualBlock(2*Config.IN_CH, 2*Config.IN_CH),

            nn.Conv2d(2*Config.IN_CH, 2*Config.IN_CH, kernel_size=3, stride=2, padding=0),
            ResidualBlock(2*Config.IN_CH, 4*Config.IN_CH),
            ResidualBlock(4*Config.IN_CH, 4*Config.IN_CH),

            nn.Conv2d(4*Config.IN_CH, 4*Config.IN_CH, kernel_size=3, stride=2, padding=0),
            ResidualBlock(4*Config.IN_CH, 4*Config.IN_CH),
            ResidualBlock(4*Config.IN_CH, 4*Config.IN_CH),
            ResidualBlock(4*Config.IN_CH, 4*Config.IN_CH),

            AttentionBlock(4*Config.IN_CH),
            ResidualBlock(4*Config.IN_CH, 4*Config.IN_CH),

            nn.GroupNorm(32, 4*Config.IN_CH),
            nn.SiLU(),

            nn.Conv2d(4*Config.IN_CH, Config.OUT_CH, kernel_size=3, padding=1),
            nn.Conv2d(Config.OUT_CH, Config.OUT_CH, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        # noise: (B, 8/2, H/8, W/8)

        for layer in self:
            if getattr(layer, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))

            x = layer(x)

        # Getting latent representation of data (mean, variance of Gaussian distribution)
        # x: (B, 8, H/8, W/8) -> mean, ln_variance: (B, 4, H/8, W/8)
        mean, ln_variance = torch.chunk(x, 2, dim=1)
        ln_variance = torch.clamp(ln_variance, -30, 20)
        variance = torch.exp(ln_variance)
        std_dev = variance ** 0.5

        # Sampling from Gaussian
        x = std_dev*noise + mean
        x *= Config.SCALING_OUTPUT_CONSTANT

        return x
