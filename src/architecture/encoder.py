import torch
import torch.nn as nn

from architecture.attention import MultiHeadAttentionLayer
from architecture.feed_forward import FFLayer
from architecture.residual_connection import ResidualConnection
from architecture.layer_normalization import LayerNorm


class EncoderBlock(nn.Module):
    """
    Transformer encoder block consisting of multi-head attention
    and feed-forward layers with residual connections.

    :param dims: The dimensionality of the input and output features.
    :type dims: int
    :param dropout_val: The dropout rate to apply after the residual connections.
    :type dropout_val: float
    :param att_layer: The multi-head attention layer.
    :type att_layer: MultiHeadAttentionLayer
    :param ff_layer: The feed-forward layer.
    :type ff_layer: FFLayer
    """

    def __init__(self,
                 dims: int,
                 dropout_val: float,
                 att_layer: MultiHeadAttentionLayer,
                 ff_layer: FFLayer):
        super().__init__()
        self.att_layer = att_layer
        self.ff_layer = ff_layer
        self.residual_connection_1 = ResidualConnection(dims, dropout_val)
        self.residual_connection_2 = ResidualConnection(dims, dropout_val)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder block.

        :param x: Input tensor of shape (B, S, D) where B is the batch size,
                    S is the sequence length, and D is the dimensionality.
        :type x: torch.Tensor
        :param mask: Mask tensor of shape (B, 1, S, S) to prevent attention
                        to certain positions.
        :type mask: torch.Tensor
        :return: Output tensor of shape (B, S, D) after applying the encoder block.
        :rtype: torch.Tensor
        """
        x = self.residual_connection_1(x, lambda z: self.att_layer(z, z, z, mask))
        x = self.residual_connection_2(x, self.ff_layer)

        return x


class Encoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder blocks.

    :param n: The number of encoder blocks.
    :type n: int
    :param dims: The dimensionality of the input and output features.
    :type dims: int
    :param n_heads: The number of attention heads in each multi-head
                        attention layer.
    :type n_heads: int
    :param dropout_val: The dropout rate to apply in the multi-head attention
                        and feed-forward layers.
    :type dropout_val: float
    """

    def __init__(self,
                 n: int,
                 dims: int,
                 n_heads: int,
                 dropout_val: float):
        super().__init__()
        encoder_blocks = []
        for _ in range(n):
            att_l = MultiHeadAttentionLayer(dims, n_heads, dropout_val)
            ff_l = FFLayer(dims, dropout_val)
            enc_block = EncoderBlock(dims, dropout_val, att_l, ff_l)
            encoder_blocks.append(enc_block)
        self.layers = nn.ModuleList(encoder_blocks)
        self.layer_norm = LayerNorm(dims)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.

        :param x: Input tensor of shape (B, S, D) where B is the batch size,
                    S is the sequence length, and D is the dimensionality.
        :type x: torch.Tensor
        :param mask: Mask tensor of shape (B, 1, S, S) to prevent attention
                        to certain positions.
        :type mask: torch.Tensor
        :return: Output tensor of shape (B, S, D) after passing through the encoder.
        :rtype: torch.Tensor
        """
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return x
