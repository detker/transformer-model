import sys
import torch
import torch.nn as nn

sys.path.append('../')
from attention import SelfAttention


class CLIPLayer(nn.Module):
    def __init__(self,
                 n_heads: int,
                 d_embd: int):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(d_embd)
        self.attention = SelfAttention(n_heads, d_embd)
        self.layer_norm_2 = nn.LayerNorm(d_embd)

        self.ff_1 = nn.Linear(d_embd, 4*d_embd)
        self.ff_2 = nn.Linear(4*d_embd, d_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, D) -> out: (B, seq_len, D)
        residual_connection = x
        x = self.layer_norm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residual_connection
        
        residual_connection = x
        x = self.layer_norm_2(x)
        x = self.ff_1(x)
        x = x * torch.sigmoid(1.702*x)  # QuickGELU
        x = self.ff_2(x)
        x += residual_connection

        return x
