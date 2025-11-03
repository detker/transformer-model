import sys
import torch
import torch.nn as nn
from torch.nn import functional as f

sys.path.append('../')
from config import CLIPConfig
from attention.self_attention import SelfAttention
from attention.cross_attention import CrossAttention


class UNetAttentionBlock(nn.Module):
    def __init__(self,
                 n_heads: int,
                 head_size: int,
                 d_context: int = CLIPConfig.EMBD_D):
        super().__init__()

        self.channels = n_heads * head_size
        self.group_norm = nn.GroupNorm(32, self.channels, eps=1e-6)
        self.conv2d = nn.Conv2d(self.channels, self.channels, kernel_size=1, padding=0)

        self.layer_norm_1 = nn.LayerNorm(self.channels)
        self.self_attention = SelfAttention(n_heads, self.channels, in_bias=False)

        self.layer_norm_2 = nn.LayerNorm(self.channels)
        self.cross_attention = CrossAttention(n_heads, self.channels, d_context, in_bias=False)

        self.layer_norm_3 = nn.LayerNorm(self.channels)

        self.ff_geglu_1 = nn.Linear(self.channels, 2 * 4 * self.channels)
        self.ff_geglu_2 = nn.Linear(4 * self.channels, self.channels)

        self.conv_out = nn.Conv2d(self.channels, self.channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # x: (B, channels, H, W)
        # context: (B, seq_len=77, d_context)

        long_residue_connection = x

        x = self.group_norm(x)
        x = self.conv2d(x)

        B, F, H, W = x.shape
        x = x.view(B, F, H * W).transpose(-1, -2)  # (B, H*W, F)

        short_residue_connection = x
        x = self.layer_norm_1(x)
        x = self.self_attention(x)
        x += short_residue_connection

        short_residue_connection = x
        x = self.layer_norm_2(x)
        x = self.cross_attention(x, context)
        x += short_residue_connection

        short_residue_connection = x
        x = self.layer_norm_3(x)
        x = self.ff_geglu_1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * f.gelu(gate)
        x = self.ff_geglu_2(x)
        x += short_residue_connection

        x = x.transpose(-1, -2).contiguous().view(B, F, H, W)
        x = self.conv_out(x)
        x = x + long_residue_connection

        return x
