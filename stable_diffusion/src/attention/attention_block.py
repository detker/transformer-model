import sys
import torch
import torch.nn as nn

sys.path.append('../')
from attention.self_attention import SelfAttention


class AttentionBlock(nn.Module):
    def __init__(self,
                 features: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, features)
        self.attention = SelfAttention(1, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in: (B, features, H, W) -> out: (B, features, H, W)
        b, f, h, w = x.shape

        residual_connection = x

        x = self.group_norm(x)

        x = x.view(b, f, h * w)
        x = x.transpose(-1, -2)  # (B, H*W, F)

        x = self.attention(x)

        x = x.transpose(-1, -2)
        x = x.view(b, f, h, w)

        x += residual_connection

        return x
