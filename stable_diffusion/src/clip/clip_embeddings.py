import torch
import torch.nn as nn


class CLIPEmbeddings(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_embd: int,
                 seq_len: int):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, d_embd)
        self.positional_embd = nn.Parameter(torch.zeros(seq_len, d_embd))

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # in: (B, seq_len) -> out: (B, seq_len, d_embd)
        x = self.embedding_layer(x)
        x += self.positional_embd

        return x
