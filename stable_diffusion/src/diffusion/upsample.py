import torch
import torch.nn as nn
from torch.nn import functional as F


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in: (B, C, H, W) -> out: (B, channels, 2*H, 2*W)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x