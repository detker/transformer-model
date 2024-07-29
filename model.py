import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over a batch of inputs.

    :param features: The number of features in the input tensor.
    :type features: int
    :param eps: A value added to the denominator for numerical stability,
                defaults to 1e-6
    :type eps: float, optional
    """

    def __init__(self,
                 features: int,
                 eps: float=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features)) # (features, )
        self.beta = nn.Parameter(torch.zeros(features)) # (features, )
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for layer normalization.

        :param x: Input tensor of shape (B, S, D) where B is the batch size, 
                    S is the sequence length, and D is the number of features.
        :type x: torch.Tensor
        :return: Layer-normalized tensor of shape (B, S, D)
        :rtype: torch.Tensor
        :raises AssertionError: If the shape of the input tensor does not match
                                the shape of the output tensor.
        """
        y = self.gamma * ((x - x.mean(dim=-1, keepdim=True)) * (
                x.var(dim=-1, keepdim=True) + self.eps)**(-0.5)) + self.beta
        assert x.shape == y.shape
        return y 
    
class FeedForwardLayer(nn.Module):
    """
    Feed-forward neural network layer used in transformer models.

    :param d_model: The dimensionality of the input and output features.
    :type d_model: int
    :param d_ff: The dimensionality of the inner feed-forward layer.
    :type d_ff: int
    :param dropout: The dropout rate to apply after 
                    the first linear transformation.
    :type dropout: float
    """

    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_layer_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass for the feed-forward layer.

        :param x: Input tensor of shape (B, S, D) where B is the batch size, 
                    S is the sequence length, and D is the number of features.
        :type x: torch.Tensor
        :return: Output tensor of shape (B, S, D) after applying the feed-forward
                    transformations and dropout.
        :rtype: torch.Tensor
        :raises AssertionError: If the shape of the input tensor does not match 
                                the shape of the output tensor.
        """
        y = self.dropout(self.linear_layer_2(self.relu(
                self.linear_layer_1(x))))
        assert x.shape == y.shape 
        return y

class InputEmbeddingLayer(nn.Module):
    """
    Embedding layer that converts input tokens to dense vectors with scaling.

    :param vocab_size: The size of the vocabulary.
    :type vocab_size: int
    :param d_model: The dimensionality of the embedding vectors.
    :type d_model: int
    """

    def __init__(self, 
                 vocab_size: int, 
                 d_model: int):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass for the embedding layer.

        :param x: Input tensor of shape (B, S) where B is the batch size 
                    and S is the sequence length.
        :type x: torch.Tensor
        :return: Output tensor of shape (B, S, D) where D is the dimensionality
                    of the embeddings.
        :rtype: torch.Tensor
        """
        y = self.emb(x) * (self.d_model)**(0.5)
        return y

class PositionalEmbeddingLayer(nn.Module):
    """
    Adds positional embeddings to the input tensor to provide positional information.

    :param context_size: The size of the context (i.e., the maximum sequence length).
    :type context_size: int
    :param d_model: The dimensionality of the embeddings.
    :type d_model: int
    :param dropout: The dropout rate to apply after adding positional embeddings.
    :type dropout: float
    """
    
    def __init__(self, 
                 context_size: int, 
                 d_model: int, 
                 dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.ones((context_size, d_model), dtype=torch.float32)
        # using x = exp(ln(x)) for numerical stability
        positions = torch.arange(0, context_size).unsqueeze(1) # (context_size, 1)
        denominator = torch.exp((-1)*torch.arange(0, d_model, 2)/d_model * 
                                math.log(10_000.0)) # (D//2, )
        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # (1, context_size, D)

    def forward(self, x):
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
        return self.dropout(y)

class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization and dropout.

    :param d_model: The dimensionality of the input and output features.
    :type d_model: int
    :param dropout: The dropout rate to apply after the residual connection.
    :type dropout: float
    """

    def __init__(self, 
                 d_model: int, 
                 dropout: float):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, layer):
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
        x = x + layer(self.norm(x))
        return self.dropout(x)

class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head attention mechanism as described in 
    "Attention is All You Need" paper.

    :param d_model: The dimensionality of the input and output features.
    :type d_model: int
    :param n_heads: The number of attention heads.
    :type n_heads: int
    :param dropout: The dropout rate to apply on the attention scores.
    :type dropout: float
    :raises AssertionError: If `d_model` is not divisible by `n_heads`.
    """

    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 dropout: float):
        assert d_model % n_heads == 0
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
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
               (self.head_size)**(-0.5)) # (B, n_heads, S, S)
        
        # applying mask (defined separately for encoder and decoder)
        if mask is not None:
            att = torch.masked_fill(att, mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ value # (B, n_heads, S, head_size)
        y = y.transpose(1, 2).contiguous().view(-1, 
                                                y.shape[2], 
                                                self.d_model) # (B, S, D)
        
        assert q.shape == self.W_o(y).shape
        return self.W_o(y) # (B, S, D)

class EncoderBlock(nn.Module):
    """
    Transformer encoder block consisting of multi-head attention 
    and feed-forward layers with residual connections.

    :param d_model: The dimensionality of the input and output features.
    :type d_model: int
    :param dropout: The dropout rate to apply after the residual connections.
    :type dropout: float
    :param att_layer: The multi-head attention layer.
    :type att_layer: MultiHeadAttentionLayer
    :param ff_layer: The feed-forward layer.
    :type ff_layer: FeedForwardLayer
    """

    def __init__(self, 
                 d_model: int, 
                 dropout: float, 
                 att_layer: MultiHeadAttentionLayer, 
                 ff_layer: FeedForwardLayer):
        super().__init__()
        self.att_layer = att_layer
        self.ff_layer = ff_layer
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, mask):
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
        x = self.residual_connections[0](
            x, lambda z: self.att_layer(z, z, z, mask)
        )
        x = self.residual_connections[1](
            x, self.ff_layer
        )
        return x

class Encoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder blocks.

    :param n: The number of encoder blocks.
    :type n: int
    :param d_model: The dimensionality of the input and output features.
    :type d_model: int
    :param n_heads: The number of attention heads in each multi-head 
                        attention layer.
    :type n_heads: int
    :param d_ff: The dimensionality of the feed-forward layer.
    :type d_ff: int
    :param dropout: The dropout rate to apply in the multi-head attention 
                        and feed-forward layers.
    :type dropout: float
    """

    def __init__(self, 
                 n: int, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int, 
                 dropout: float):
        super().__init__()
        encoder_blocks = []
        for _ in range(n):
            att_l = MultiHeadAttentionLayer(d_model, n_heads, dropout)
            ff_l = FeedForwardLayer(d_model, d_ff, dropout)
            enc_block = EncoderBlock(d_model, dropout, att_l, ff_l)
            encoder_blocks.append(enc_block)
        self.layers = nn.ModuleList(encoder_blocks)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
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
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    Transformer decoder block consisting of self-attention, cross-attention, 
    and feed-forward layers with residual connections.

    :param d_model: The dimensionality of the input and output features.
    :type d_model: int
    :param dropout: The dropout rate to apply after the residual connections.
    :type dropout: float
    :param self_att_layer: The multi-head self-attention layer.
    :type self_att_layer: MultiHeadAttentionLayer
    :param cross_att_layer: The multi-head cross-attention layer.
    :type cross_att_layer: MultiHeadAttentionLayer
    :param ff_layer: The feed-forward layer.
    :type ff_layer: FeedForwardLayer
    """

    def __init__(self, 
                 d_model: int, 
                 dropout: float, 
                 self_att_layer: MultiHeadAttentionLayer, 
                 cross_att_layer: MultiHeadAttentionLayer, 
                 ff_layer: FeedForwardLayer):
        super().__init__()
        self.self_att_layer = self_att_layer
        self.cross_att_layer = cross_att_layer
        self.ff_layer = ff_layer
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(3)]
        )

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
        x = self.residual_connections[0](
            x, lambda z: self.self_att_layer(z, z, z, target_mask)
        )
        x = self.residual_connections[1](
            x, lambda z: self.cross_att_layer(z, enc_output, enc_output, source_mask)
        )
        x = self.residual_connections[2](
            x, self.ff_layer
        )
        return x

class Decoder(nn.Module):
    """
    Transformer decoder consisting of multiple decoder blocks.

    :param n: The number of decoder blocks.
    :type n: int
    :param d_model: The dimensionality of the input and output features.
    :type d_model: int
    :param n_heads: The number of attention heads in each multi-head attention layer.
    :type n_heads: int
    :param d_ff: The dimensionality of the feed-forward layer.
    :type d_ff: int
    :param dropout: The dropout rate to apply in the attention and feed-forward layers.
    :type dropout: float
    """

    def __init__(self, 
                 n: int, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int, 
                 dropout: float):
        super().__init__()
        decoder_blocks = []
        for _ in range(n):
            self_att_l = MultiHeadAttentionLayer(d_model, n_heads, dropout)
            cross_att_l = MultiHeadAttentionLayer(d_model, n_heads, dropout)
            ff_l = FeedForwardLayer(d_model, d_ff, dropout)
            dec_block = DecoderBlock(d_model, dropout, self_att_l, 
                                     cross_att_l, ff_l)
            decoder_blocks.append(dec_block)
        self.layers = nn.ModuleList(decoder_blocks)
        self.norm = LayerNormalization(d_model)

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
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Linear projection layer that maps the model's output to the vocabulary space.

    :param d_model: The dimensionality of the input features.
    :type d_model: int
    :param vocab_size: The size of the vocabulary, i.e., 
                        the dimensionality of the output space.
    :type vocab_size: int
    """
    def __init__(self, 
                 d_model: int, 
                 vocab_size: int):
        super().__init__()
        self.proj_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Forward pass for the projection layer.

        :param x: Input tensor of shape (B, S, D) where B is the batch size, 
                    S is the sequence length, and D is the dimensionality.
        :type x: torch.Tensor
        :return: Output tensor of shape (B, S, vocab_size) after 
                    applying the linear transformation.
        :rtype: torch.Tensor
        """
        # in: (B, S, D) -> out: (B, S, vocab_size)
        return self.proj_layer(x)

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
    :param d_model: The dimensionality of the model's hidden layers (default is 512).
    :type d_model: int, optional
    :param n_heads: The number of attention heads in each 
                    multi-head attention layer (default is 8).
    :type n_heads: int, optional
    :param d_ff: The dimensionality of the feed-forward layer (default is 2048).
    :type d_ff: int, optional
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
                 d_model: int = 512, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 n: int = 6, 
                 dropout: int = 0.2):
        super().__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_context_size = source_context_size
        self.target_context_size = target_context_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n = n
        self.dropout = dropout
        self.encoder = Encoder(n, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n, d_model, n_heads, d_ff, dropout)
        self.projector = ProjectionLayer(d_model, target_vocab_size)
        self.source_embeddings = InputEmbeddingLayer(source_vocab_size, d_model)
        self.source_pos_embeddings = PositionalEmbeddingLayer(source_context_size, 
                                                              d_model, dropout)
        self.target_embeddings = InputEmbeddingLayer(target_vocab_size, d_model)
        self.target_pos_embeddings = PositionalEmbeddingLayer(target_context_size, 
                                                              d_model, dropout)

    def encode(self, x, src_mask):
        """
        Perform encoding on the source sequence.

        :param x: Source input tensor of shape (B, S) 
                    where B is the batch size and S is the sequence length.
        :type x: torch.Tensor
        :param src_mask: Mask tensor for the source sequence of shape (B, 1, S, S).
        :type src_mask: torch.Tensor
        :return: Encoded output tensor of shape (B, S, D).
        :rtype: torch.Tensor
        """
        x = self.source_embeddings(x)
        x = self.source_pos_embeddings(x)
        return self.encoder(x, src_mask)

    def decode(self, x, enc_output, src_mask, tgt_mask):
        """
        Perform decoding on the target sequence.

        :param x: Target input tensor of shape (B, T) 
                    where B is the batch size and T is the target sequence length.
        :type x: torch.Tensor
        :param enc_output: Output tensor from the encoder of shape (B, S, D).
        :type enc_output: torch.Tensor
        :param src_mask: Mask tensor for the source sequence of shape (B, 1, S, S).
        :type src_mask: torch.Tensor
        :param tgt_mask: Mask tensor for the target sequence of shape (B, 1, T, T).
        :type tgt_mask: torch.Tensor
        :return: Decoded output tensor of shape (B, T, D).
        :rtype: torch.Tensor
        """
        x = self.target_embeddings(x)
        x = self.target_pos_embeddings(x)
        return self.decoder(x, enc_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Project the decoder output to the target vocabulary space.

        :param x: Decoder output tensor of shape (B, T, D).
        :type x: torch.Tensor
        :return: Projected output tensor of shape (B, T, target_vocab_size).
        :rtype: torch.Tensor
        """
        return self.projector(x)

    def forward(self, x_source, x_target, src_mask, tgt_mask):
        """
        Forward pass for the transformer model.

        :param x_source: Source input tensor of shape (B, S).
        :type x_source: torch.Tensor
        :param x_target: Target input tensor of shape (B, T).
        :type x_target: torch.Tensor
        :param src_mask: Mask tensor for the source sequence of shape (B, 1, S, S).
        :type src_mask: torch.Tensor
        :param tgt_mask: Mask tensor for the target sequence of shape (B, 1, T, T).
        :type tgt_mask: torch.Tensor
        :return: Final output tensor of shape (B, T, target_vocab_size).
        :rtype: torch.Tensor
        """
        enc_out = self.encode(x_source, src_mask)
        dec_out = self.decode(x_target, enc_out, src_mask, tgt_mask)
        return self.project(dec_out)

    @staticmethod
    def build_transformer(config):
        """
        Build a Transformer model from a configuration object.

        :param config: Config object containing model parameters.
        :type config: Config
        :return: A Transformer model with initialized parameters.
        :rtype: Transformer
        """
        transformer = Transformer(config.source_vocab_size,
                        config.target_vocab_size,
                        config.sequence_length,
                        config.sequence_length,
                        config.d_model,
                        config.n_heads,
                        config.d_ff,
                        config.n,
                        config.dropout)

        for param in transformer.parameters():
            if param.dim() >= 2: nn.init.xavier_uniform_(param)

        return transformer
