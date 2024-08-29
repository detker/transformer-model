import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head attention mechanism as described in
    "Attention is All You Need" paper.

    :param dims: The dimensionality of the input and output features.
    :type dims: int
    :param n_heads: The number of attention heads.
    :type n_heads: int
    :param dropout_val: The dropout rate to apply on the attention scores.
    :type dropout_val: float
    :raises AssertionError: If `dims` is not divisible by `n_heads`.
    """

    def __init__(self,
                 dims: int,
                 n_heads: int,
                 dropout_val: float):
        assert dims % n_heads == 0
        super().__init__()
        self.d_model = dims
        self.n_heads = n_heads
        self.head_size = dims // n_heads
        self.dropout = nn.Dropout(dropout_val)

        self.W_q = nn.Linear(dims, dims, bias=False)
        self.W_k = nn.Linear(dims, dims, bias=False)
        self.W_v = nn.Linear(dims, dims, bias=False)
        self.W_o = nn.Linear(dims, dims, bias=False)

    def forward(self, q, k, v, mask):
        """
        Forward pass for the multi-head attention layer.

        :param q: Query tensor of shape (B, S, D) where B is the batch size,
                    S is the sequence length, and D is the dimensionality.
        :type q: torch.Tensor
        :param k: Key tensor of shape (B, S, D).
        :type k: torch.Tensor
        :param v: Value tensor of shape (B, S, D).
        :type v: torch.Tensor
        :param mask: Optional mask tensor of shape (B, 1, S, S) to prevent
                        attention to certain positions, defaults to None.
        :type mask: torch.Tensor, optional
        :return: Output tensor of shape (B, S, D) after applying
                    multi-head attention.
        :rtype: torch.Tensor
        :raises AssertionError: If the shape of the query tensor does not
                                    match the shape of the output tensor.
        """
        # (B, S, D)
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        # (B, n_heads, S, head_size)
        query = query.view(-1, query.shape[1], self.n_heads,
                           self.head_size).transpose(1, 2)
        key = key.view(-1, key.shape[1], self.n_heads,
                       self.head_size).transpose(1, 2)
        value = value.view(-1, value.shape[1], self.n_heads,
                           self.head_size).transpose(1, 2)

        att = ((query @ key.transpose(-1, -2)) *
               (self.head_size) ** (-0.5))  # (B, n_heads, S, S)

        # applying mask (defined separately for encoder and decoder)
        if mask is not None:
            att = torch.masked_fill(att, mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ value  # (B, n_heads, S, head_size)
        y = y.transpose(1, 2).contiguous().view(-1,
                                                y.shape[2],
                                                self.d_model)  # (B, S, D)

        assert q.shape == self.W_o(y).shape
        y = self.W_o(y)  # (B, S, D)
        return y
