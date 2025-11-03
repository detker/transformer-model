import torch
import torch.nn as nn
from torch.nn import functional as F


class UNetOutputLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_features)
        self.conv2d = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in: (B, 320, H/8, W/8) -> out: (B, 4, H/8, W/8):
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv2d(x)

        return x
