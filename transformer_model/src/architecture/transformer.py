import torch
import torch.nn as nn

from architecture.input_embedding import InputEmbeddingLayer
from architecture.input_embedding import PositionalEmbeddingLayer
from architecture.encoder import Encoder
from architecture.decoder import Decoder
from architecture.output_layer import ProjOutputLayer


class Transformer(nn.Module):
    """
    A Transformer model for sequence-to-sequence tasks,
    including encoding and decoding with positional and input embeddings.

    :param source_vocab_size: The size of the source vocabulary.
    :type source_vocab_size: int
    :param target_vocab_size: The size of the target vocabulary.
    :type target_vocab_size: int
    :param source_context_size: The maximum context size (sequence length)
                                for the source input.
    :type source_context_size: int
    :param target_context_size: The maximum context size (sequence length)
                                for the target input.
    :type target_context_size: int
    :param dims: The dimensionality of the model's hidden layers (default is 512).
    :type dims: int, optional
    :param n_heads: The number of attention heads in each
                    multi-head attention layer (default is 8).
    :type n_heads: int, optional
    :param n: The number of encoder and decoder blocks (default is 6).
    :type n: int, optional
    :param dropout: The dropout rate applied in the attention
                    and feed-forward layers (default is 0.2).
    :type dropout: float, optional
    """

    def __init__(self,
                 source_vocab_size: int,
                 target_vocab_size: int,
                 source_context_size: int,
                 target_context_size: int,
                 dims: int = 512,
                 n_heads: int = 8,
                 n: int = 6,
                 dropout: int = 0.2):
        super().__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_context_size = source_context_size
        self.target_context_size = target_context_size
        self.d_model = dims
        self.n_heads = n_heads
        self.n = n
        self.dropout = dropout
        self.encoder = Encoder(n, dims, n_heads, dropout)
        self.decoder = Decoder(n, dims, n_heads, dropout)
        self.projector = ProjOutputLayer(dims, target_vocab_size)
        self.source_embeddings = InputEmbeddingLayer(source_vocab_size, dims)
        self.source_pos_embeddings = PositionalEmbeddingLayer(source_context_size, dims, dropout)
        self.target_embeddings = InputEmbeddingLayer(target_vocab_size, dims)
        self.target_pos_embeddings = PositionalEmbeddingLayer(target_context_size, dims, dropout)

    def encode(self, x, source_mask):
        """
        Perform encoding on the source sequence.

        :param x: Source input tensor of shape (B, S)
                    where B is the batch size and S is the sequence length.
        :type x: torch.Tensor
        :param source_mask: Mask tensor for the source sequence of shape (B, 1, S, S).
        :type source_mask: torch.Tensor
        :return: Encoded output tensor of shape (B, S, D).
        :rtype: torch.Tensor
        """
        x = self.source_embeddings(x)
        x = self.source_pos_embeddings(x)
        x = self.encoder(x, source_mask)
        return x

    def decode(self, x, enc_output, source_mask, target_mask):
        """
        Perform decoding on the target sequence.

        :param x: Target input tensor of shape (B, T)
                    where B is the batch size and T is the target sequence length.
        :type x: torch.Tensor
        :param enc_output: Output tensor from the encoder of shape (B, S, D).
        :type enc_output: torch.Tensor
        :param source_mask: Mask tensor for the source sequence of shape (B, 1, S, S).
        :type source_mask: torch.Tensor
        :param target_mask: Mask tensor for the target sequence of shape (B, 1, T, T).
        :type target_mask: torch.Tensor
        :return: Decoded output tensor of shape (B, T, D).
        :rtype: torch.Tensor
        """
        x = self.target_embeddings(x)
        x = self.target_pos_embeddings(x)
        return self.decoder(x, enc_output, source_mask, target_mask)

    def project(self, x):
        """
        Project the decoder output to the target vocabulary space.

        :param x: Decoder output tensor of shape (B, T, D).
        :type x: torch.Tensor
        :return: Projected output tensor of shape (B, T, target_vocab_size).
        :rtype: torch.Tensor
        """
        x = self.projector(x)
        return x

    def forward(self, x_source, x_target, source_mask, target_mask):
        """
        Forward pass for the transformer model.

        :param x_source: Source input tensor of shape (B, S).
        :type x_source: torch.Tensor
        :param x_target: Target input tensor of shape (B, T).
        :type x_target: torch.Tensor
        :param source_mask: Mask tensor for the source sequence of shape (B, 1, S, S).
        :type source_mask: torch.Tensor
        :param target_mask: Mask tensor for the target sequence of shape (B, 1, T, T).
        :type target_mask: torch.Tensor
        :return: Final output tensor of shape (B, T, target_vocab_size).
        :rtype: torch.Tensor
        """
        enc_out = self.encode(x_source, source_mask)
        dec_out = self.decode(x_target, enc_out, source_mask, target_mask)
        out = self.project(dec_out)
        return out

    @staticmethod
    def build(config):
        """
        Build a Transformer model from a configuration object.

        :param config: Config object containing model parameters.
        :type config: Config
        :return: A Transformer model with initialized parameters.
        :rtype: Transformer
        """
        transformer = Transformer(config.source_vocab_size, config.target_vocab_size, config.sequence_length,
                                  config.sequence_length, config.dims, config.n_heads, config.n, config.dropout)

        for param in transformer.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param)

        return transformer
