import torch


class Utils():
    @staticmethod
    def get_triu_mask(n):
        """
        Creates an upper triangular mask tensor with a specified size.

        This mask is used in attention mechanisms to prevent the model
        from attending to future tokens. The mask has the same size in both
        dimensions and is filled with 0s in the upper triangular part
        (excluding the diagonal) and 1s elsewhere.

        :param n: The size of the mask, which corresponds to the sequence length.
        :type n: int
        :return: An upper triangular mask tensor with shape `(1, size, size)`.
                 The tensor has 0s in the upper triangular part (excluding the diagonal)
                 and 1s elsewhere.
        :rtype: torch.Tensor
        """
        mask = torch.triu(
            torch.ones((1, n, n), dtype=torch.int32), diagonal=1
        ) == 0  # (1, size, size)
        return mask
