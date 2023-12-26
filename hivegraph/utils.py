import os
import random

import numpy as np
import torch
from torch_geometric.data import Data


__all__ = ["set_seed"]


def set_seed(seed: int = 42) -> None:
    """
    Utility Function to set seeds for reproducibility

    Args:
        seed(int): Random seed, defaults to 42
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def num_graphs(data: Data) -> int:
    """
    Utility function to return the number of graphs in a data object

    Args:
        data(Data): Data object

    Returns:
        Number of graphs in the data object
    """
    if hasattr(data, "num_graphs"):
        return data.num_graphs
    else:
        return data.x.size(0)
