import torch
import torch.nn as nn

from architecture.attention import MultiHeadAttentionLayer
from architecture.feed_forward import FFLayer
from architecture.residual_connection import ResidualConnection
from architecture.layer_normalization import LayerNorm


class DecoderBlock(nn.Module):
    """
    Transformer decoder block consisting of self-attention, cross-attention,
    and feed-forward layers with residual connections.

    :param dims: The dimensionality of the input and output features.
    :type dims: int
    :param dropout_val: The dropout rate to apply after the residual connections.
    :type dropout_val: float
    :param self_att_layer: The multi-head self-attention layer.
    :type self_att_layer: MultiHeadAttentionLayer
    :param cross_att_layer: The multi-head cross-attention layer.
    :type cross_att_layer: MultiHeadAttentionLayer
    :param ff_layer: The feed-forward layer.
    :type ff_layer: FFLayer
    """

    def __init__(self,
                 dims: int,
                 dropout_val: float,
                 self_att_layer: MultiHeadAttentionLayer,
                 cross_att_layer: MultiHeadAttentionLayer,
                 ff_layer: FFLayer):
        super().__init__()
        self.self_att_layer = self_att_layer
        self.cross_att_layer = cross_att_layer
        self.ff_layer = ff_layer
        self.residual_connection_1 = ResidualConnection(dims, dropout_val)
        self.residual_connection_2 = ResidualConnection(dims, dropout_val)
        self.residual_connection_3 = ResidualConnection(dims, dropout_val)

    def forward(self, x, enc_output, source_mask, target_mask):
        """
        Forward pass for the decoder block.

        :param x: Input tensor of shape (B, T, D) where B is the batch size,
                    T is the target sequence length, and D is the dimensionality.
        :type x: torch.Tensor
        :param enc_output: Encoder output tensor of shape (B, S, D)
                            where S is the source sequence length.
        :type enc_output: torch.Tensor
        :param source_mask: Source mask tensor of shape (B, 1, S, S) to prevent
                                attention to certain positions in the source sequence.
        :type source_mask: torch.Tensor
        :param target_mask: Target mask tensor of shape (B, 1, T, T) to prevent
                                attention to certain positions in the target sequence.
        :type target_mask: torch.Tensor
        :return: Output tensor of shape (B, T, D) after passing through the decoder block.
        :rtype: torch.Tensor
        """
        x = self.residual_connection_1(x, lambda z: self.self_att_layer(z, z, z, target_mask))
        x = self.residual_connection_2(x, lambda z: self.cross_att_layer(z, enc_output, enc_output, source_mask))
        x = self.residual_connection_3(x, self.ff_layer)

        return x


class Decoder(nn.Module):
    """
    Transformer decoder consisting of multiple decoder blocks.

    :param n: The number of decoder blocks.
    :type n: int
    :param dims: The dimensionality of the input and output features.
    :type dims: int
    :param n_heads: The number of attention heads in each multi-head attention layer.
    :type n_heads: int
    :param dropout_val: The dropout rate to apply in the attention and feed-forward layers.
    :type dropout_val: float
    """

    def __init__(self,
                 n: int,
                 dims: int,
                 n_heads: int,
                 dropout_val: float):
        super().__init__()
        decoder_blocks = []
        for _ in range(n):
            self_att_l = MultiHeadAttentionLayer(dims, n_heads, dropout_val)
            cross_att_l = MultiHeadAttentionLayer(dims, n_heads, dropout_val)
            ff_l = FFLayer(dims, dropout_val)
            dec_block = DecoderBlock(dims, dropout_val, self_att_l, cross_att_l, ff_l)
            decoder_blocks.append(dec_block)
        self.layers = nn.ModuleList(decoder_blocks)
        self.norm = LayerNorm(dims)

    def forward(self, x, enc_output, source_mask, target_mask):
        """
        Forward pass for the decoder.

        :param x: Input tensor of shape (B, T, D) where B is the batch size,
                    T is the target sequence length, and D is the dimensionality.
        :type x: torch.Tensor
        :param enc_output: Encoder output tensor of shape (B, S, D)
                            where S is the source sequence length.
        :type enc_output: torch.Tensor
        :param source_mask: Source mask tensor of shape (B, 1, S, S) to prevent
                                attention to certain positions in the source sequence.
        :type source_mask: torch.Tensor
        :param target_mask: Target mask tensor of shape (B, 1, T, T) to prevent
                                attention to certain positions in the target sequence.
        :type target_mask: torch.Tensor
        :return: Output tensor of shape (B, T, D) after passing through the decoder.
        :rtype: torch.Tensor
        """
        for layer in self.layers:
            x = layer(x, enc_output, source_mask, target_mask)
        x = self.norm(x)
        return x
