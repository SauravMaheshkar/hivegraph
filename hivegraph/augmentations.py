from typing import List

import torch


__all__: List[str] = ["drop_feature"]


def drop_feature(
    x: torch.Tensor, drop_prob: float, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Implementation of Masking Node Features (MF)

    Randomly masks a fraction of dimensions with zeros in node features.

    Ref: Section 3.2.2 of https://arxiv.org/abs/2006.04131v2

    Args:
        x (torch.Tensor): Node features.
        drop_prob (float): Probability of dropping a feature.
        dtype (torch.dtype): Data type of the tensor.

    Returns:
        torch.Tensor: Masked node features.
    """
    drop_mask: torch.Tensor = (
        torch.empty((x.size(1),), dtype=dtype, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0

    return x
