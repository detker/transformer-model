import sys
import torch
import torch.nn as nn

sys.path.append('../')
from config import VAEEncoderConfig as EncConfig
from config import VAEDEcoderConfig as Config
from VAE.residual_block import ResidualBlock
from attention import AttentionBlock


class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
           nn.Conv2d(EncConfig.OUT_CH//2, EncConfig.OUT_CH//2, kernel_size=1, padding=0),
           nn.Conv2d(EncConfig.OUT_CH//2, Config.IN_CH, kernel_size=3, padding=1),

           ResidualBlock(Config.IN_CH, Config.IN_CH),

           AttentionBlock(Config.IN_CH),

           ResidualBlock(Config.IN_CH, Config.IN_CH),
           ResidualBlock(Config.IN_CH, Config.IN_CH),
           ResidualBlock(Config.IN_CH, Config.IN_CH),
           ResidualBlock(Config.IN_CH, Config.IN_CH),

           nn.Upsample(scale_factor=2),

           nn.Conv2d(Config.IN_CH, Config.IN_CH, kernel_size=3, padding=1),

           ResidualBlock(Config.IN_CH, Config.IN_CH),
           ResidualBlock(Config.IN_CH, Config.IN_CH),
           ResidualBlock(Config.IN_CH, Config.IN_CH),

           nn.Upsample(scale_factor=2),

           nn.Conv2d(Config.IN_CH, Config.IN_CH, kernel_size=3, padding=1),

           ResidualBlock(Config.IN_CH, Config.IN_CH//2),
           ResidualBlock(Config.IN_CH//2, Config.IN_CH//2),
           ResidualBlock(Config.IN_CH//2, Config.IN_CH//2),

           nn.Upsample(scale_factor=2),

           nn.Conv2d(Config.IN_CH//2, Config.IN_CH//2, kernel_size=3, padding=1),

           ResidualBlock(Config.IN_CH//2, Config.IN_CH//4),
           ResidualBlock(Config.IN_CH//4, Config.IN_CH//4),
           ResidualBlock(Config.IN_CH//4, Config.IN_CH//4),

           nn.GroupNorm(32, Config.IN_CH//4),

           nn.SiLU(),

           nn.Conv2d(Config.IN_CH//4, EncConfig.IMG_CH, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in: (B, 4, H/8, W/8) -> out: (B, 3, H, W)

        x /= EncConfig.SCALING_OUTPUT_CONSTANT

        for layer in self:
            x = layer(x)

        return x
