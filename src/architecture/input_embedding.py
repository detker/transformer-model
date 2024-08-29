import math
import torch
import torch.nn as nn


class InputEmbeddingLayer(nn.Module):
    """
    Embedding layer that converts input tokens to dense vectors with scaling.

    :param vocab_size: The size of the vocabulary.
    :type vocab_size: int
    :param dims: The dimensionality of the embedding vectors.
    :type dims: int
    """

    def __init__(self,
                 vocab_size: int,
                 dims: int):
        super().__init__()

        self.dims = dims
        self.emb = nn.Embedding(vocab_size, dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        :param x: Input tensor of shape (B, S) where B is the batch size
                    and S is the sequence length.
        :type x: torch.Tensor
        :return: Output tensor of shape (B, S, D) where D is the dimensionality
                    of the embeddings.
        :rtype: torch.Tensor
        """
        y = self.emb(x) * (self.dims) ** (0.5)
        return y


class PositionalEmbeddingLayer(nn.Module):
    """
    Adds positional embeddings to the input tensor to provide positional information.

    :param context_size: The size of the context (i.e., the maximum sequence length).
    :type context_size: int
    :param dims: The dimensionality of the embeddings.
    :type dims: int
    :param dropout_val: The dropout rate to apply after adding positional embeddings.
    :type dropout_val: float
    """

    def __init__(self,
                 context_size: int,
                 dims: int,
                 dropout_val: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout_val)

        positions = torch.arange(context_size, dtype=torch.float32)[:, None]  # (context_size, 1)
        denominator = torch.exp(torch.arange(0, dims, 2) * -(math.log(10000.0) / dims))

        pe = torch.empty(context_size, dims)
        pe[:, 0::2] = torch.sin(positions * denominator)  # sin values for even indices
        pe[:, 1::2] = torch.cos(positions * denominator)  # cos values for odd indices

        self.register_buffer('pe', pe[None])  # (1, context_size, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the positional embedding layer.

        :param x: Input tensor of shape (B, S, D) where B is the batch size,
                    S is the sequence length, and D is the number of features.
        :type x: torch.Tensor
        :return: Output tensor of shape (B, S, D) with positional embeddings
                    added and dropout applied.
        :rtype: torch.Tensor
        :raises AssertionError: If the shape of the input tensor does not match
                                    the shape of the output tensor.
        """
        y = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        assert x.shape == y.shape
        y = self.dropout(y)

        return y
