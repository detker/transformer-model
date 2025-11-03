import torch
import torch.nn as nn


class ProjOutputLayer(nn.Module):
    """
    Linear projection layer that maps the model's output to the vocabulary space.

    :param dims: The dimensionality of the input features.
    :type dims: int
    :param vocab_size: The size of the vocabulary, i.e.,
                        the dimensionality of the output space.
    :type vocab_size: int
    """

    def __init__(self,
                 dims: int,
                 vocab_size: int):
        super().__init__()
        self.proj_layer = nn.Linear(dims, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.proj_layer(x)
        return x
