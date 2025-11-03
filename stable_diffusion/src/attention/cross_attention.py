import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int,
                 dims: int,
                 d_context: int,
                 in_bias: bool = True,
                 out_bias: bool = True):
        super().__init__()

        self.head_size = dims // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(dims, dims, bias=in_bias)
        self.w_k = nn.Linear(d_context, dims, bias=in_bias)
        self.w_v = nn.Linear(d_context, dims, bias=in_bias)
        self.w_o = nn.Linear(dims, dims, bias=out_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x - latent: (B, seq_len_Q, D_Q)
        # context: (B, seq_len_KV, D_KV)

        B, seq_len_Q, D_Q = x.shape
        
        q = self.w_q(x)
        k = self.w_k(context)
        v = self.w_v(context)

        q = q.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, seq_len_Q, head_size)
        k = k.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, seq_len_Q, head_size)
        v = v.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, seq_len_Q, head_size)

        weights = q @ k.transpose(-1, -2)  # (B, n_heads, seq_len_Q, seq_len_Q)
        weights = weights / (self.head_size**0.5)
        weights = F.softmax(weights, dim=-1)

        out = weights @ v  # (B, n_heads, seq_len_Q, D_Q)
        out = out.transpose(1, 2).contiguous().view(B, seq_len_Q, D_Q)
        out = self.w_o(out)

        return out
        