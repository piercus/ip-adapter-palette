import torch
from torch import Tensor, diag, matmul, mean, unsqueeze, exp

def mmd(x: Tensor, y: Tensor, sigma: int = 10, scale: int = 1) -> Tensor:
    """Memory-efficient MMD implementation.

    Inpired from https://github.com/sayakpaul/cmmd-pytorch/blob/main/distance.py
    and https://github.com/sayakpaul/cmmd-pytorch/

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """

    x_sqnorms = diag(matmul(x, x.T))
    y_sqnorms = diag(matmul(y, y.T))

    gamma = 1 / (2 * sigma**2)
    k_xx = mean(
        exp(-gamma * (-2 * matmul(x, x.T) + unsqueeze(x_sqnorms, 1) + unsqueeze(x_sqnorms, 0)))
    )
    k_xy = mean(
        exp(-gamma * (-2 * matmul(x, y.T) + unsqueeze(x_sqnorms, 1) + unsqueeze(y_sqnorms, 0)))
    )
    k_yy = mean(
        exp(-gamma * (-2 * matmul(y, y.T) + unsqueeze(y_sqnorms, 1) + unsqueeze(y_sqnorms, 0)))
    )

    return scale * (k_xx + k_yy - 2 * k_xy)