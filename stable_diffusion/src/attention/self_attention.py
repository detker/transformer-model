import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int,
                 dims: int,
                 in_bias=True,
                 out_bias=True):
        super().__init__()

        self.w_qkv = nn.Linear(dims, 3 * dims, bias=in_bias)
        self.w_o = nn.Linear(dims, dims, bias=out_bias)
        self.n_heads = n_heads
        self.head_size = dims // n_heads
        self.dims = dims

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x: (B, seq_len, D)
        B, seq_len, D = x.shape
        q, k, v = torch.chunk(self.w_qkv(x), 3, dim=-1)

        # (B, seq_len, D) -> (B, n_heads, seq_len, head_size)
        q = q.view(B, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, seq_len, self.n_heads, self.head_size).transpose(1, 2)

        # (B, n_heads, seq_len, seq_len)
        weights = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = (torch.ones((seq_len, seq_len), dtype=torch.bool)
                    .triu(1).unsqueeze(0).unsqueeze(0).to(weights.device))  # (1, 1, seq_len, seq_len)
            weights.masked_fill_(mask, -torch.inf)  

        weights = weights / (self.head_size ** 0.5)
        weights = F.softmax(weights, dim=-1)

        # (B, n_heads, seq_len, head_size)
        out = weights @ v

        # (B, seq_len, D)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, D)

        out = self.w_o(out)

        return out
