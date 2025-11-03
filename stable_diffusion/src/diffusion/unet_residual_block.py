import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append('../')
from config import DiffusionConfig as Config


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time: int = 4 * Config.D_TIME):
        super().__init__()
        self.group_norm_f = nn.GroupNorm(32, in_channels)
        self.conv2d_f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.linear_time = nn.Linear(n_time, out_channels)

        self.group_norm_merged = nn.GroupNorm(32, out_channels)
        self.conv2d_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_layer = None
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # in: x:(B, in_ch, H, W), time: (1, 1280) -> out: (B, out_ch, H, W)
        residue_connection = x

        x = self.group_norm_f(x)
        x = F.silu(x)
        x = self.conv2d_f(x)

        time = F.silu(time)
        time = self.linear_time(time)

        time = time.unsqueeze(-1).unsqueeze(-1)  # (1, out_ch, 1, 1)
        x += time

        x = self.group_norm_merged(x)
        x = F.silu(x)
        x = self.conv2d_merged(x)

        x += self.residual_layer(residue_connection)

        return x
