from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


__all__: List[str] = ["NormalizedDegree"]


class NormalizedDegree(BaseTransform):
    r"""Normalizes the degree of nodes in :obj:`data` to lie in :math:`[-1, 1]`.

    Args:
        mean (float): The mean of the degree distribution.
        std (float): The standard deviation of the degree distribution.
    """

    def __init__(self, mean: float, std: float) -> None:
        """
        Initializes the transform.

        Args:
            mean (float): The mean of the degree distribution.
            std (float): The standard deviation of the degree distribution.
        """
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        """
        Performs the transform.

        Args:
            data (Data): The graph data object.

        Returns:
            Data: The transformed graph data object.
        """
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

    def __repr__(self) -> str:
        """
        Returns a string representation of the transform.

        Returns:
            str: String representation of the transform.
        """
        return "{}(mean={:.4f}, std={:.4f})".format(
            self.__class__.__name__, self.mean, self.std
        )
