import os
import random

import numpy as np
import torch

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
