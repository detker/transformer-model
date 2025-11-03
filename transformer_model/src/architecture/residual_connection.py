import torch
import torch.nn as nn

from architecture.layer_normalization import LayerNorm


class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization and dropout.

    :param dims: The dimensionality of the input and output features.
    :type dims: int
    :param dropout_val: The dropout rate to apply after the residual connection.
    :type dropout_val: float
    """

    def __init__(self,
                 dims: int,
                 dropout_val: float):
        super().__init__()

        self.layer_norm = LayerNorm(dims)
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x: torch.Tensor, layer) -> torch.Tensor:
        """
        Forward pass for the residual connection layer.

        :param x: Input tensor of shape (B, S, D) where B is the batch size,
                    S is the sequence length, and D is the number of features.
        :type x: torch.Tensor
        :param layer: The layer to apply after normalization, which should
                        also return a tensor of shape (B, S, D).
        :type layer: callable
        :return: Output tensor of shape (B, S, D) after applying the residual
                    connection, normalization, and dropout.
        :rtype: torch.Tensor
        """
        residue_connection = x
        x = self.layer_norm(x)
        x = layer(x)
        x += residue_connection
        x = self.dropout(x)

        return x
