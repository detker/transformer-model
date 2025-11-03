import torch
import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()

        self.group_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_layer = None
        if in_channels == out_channels: self.residual_layer = nn.Identity()
        else: self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        residue_connection = x

        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv2d_1(x)
        
        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv2d_2(x)

        x = x + self.residual_layer(residue_connection)

        return x
