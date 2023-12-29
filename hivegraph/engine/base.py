from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import torch


__all__: List[str] = ["BaseTrainer"]


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    """

    @abstractmethod
    def fit(
        self,
        epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        verbose: bool,
        log_to_wandb: bool,
    ) -> None:
        pass

    @abstractmethod
    def train_step(
        self, train_dataloder: Iterable, optimizer: torch.optim.Optimizer
    ) -> float:
        pass

    @abstractmethod
    def val_step(self, val_dataloder: Iterable) -> float:
        pass

    @abstractmethod
    def test_step(self, test_dataloder: Iterable, test_metric: Optional[str]) -> float:
        pass
