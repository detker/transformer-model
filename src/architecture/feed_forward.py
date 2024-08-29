import torch
import torch.nn as nn


class FFLayer(nn.Module):
    """
    Feed-forward neural network layer used in transformer models.

    :param dims: The dimensionality of the input and output features.
    :type dims: int
    :param dropout_val: The dropout rate to apply after
                    the first linear transformation.
    :type dropout_val: float
    """

    def __init__(self,
                 dims: int,
                 dropout_val: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout_val)
        self.linear_layer_1 = nn.Linear(dims, 4*dims)
        self.relu = nn.ReLU()
        self.linear_layer_2 = nn.Linear(4*dims, dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
