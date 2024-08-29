import torch
import torch.nn as nn


class LayerNorm(nn.Module):
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

        self.gamma = nn.Parameter(torch.ones(features))  # (features, )
        self.beta = nn.Parameter(torch.zeros(features))  # (features, )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
