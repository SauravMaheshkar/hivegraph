from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

__all__: List[str] = ["NormalizedDegree"]


class NormalizedDegree(BaseTransform):
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

    def __repr__(self) -> str:
        return "{}(mean={:.4f}, std={:.4f})".format(
            self.__class__.__name__, self.mean, self.std
        )
