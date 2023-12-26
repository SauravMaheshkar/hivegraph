from typing import Dict, Iterable, List, Optional

import torch
import wandb
from rich import print
from rich.progress import track
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader

from hivegraph.engine.base import BaseTrainer
from hivegraph.utils import num_graphs


__all__: List[str] = ["TransductiveTrainer"]


class TransductiveTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.random_state = random_state

    def fit(
        self,
        epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        verbose: bool,
        log_to_wandb: bool,
    ) -> None:
        train_losses: List[float] = []

        train_dataloder = get_dataloader(self.dataset, batch_size)
        self.model.to(self.device).reset_parameters()

        for epoch in track(range(1, epochs + 1)):
            train_loss: float = self.train_step(
                train_dataloder=train_dataloder, optimizer=optimizer
            )

            epoch_stats: Dict[str, float] = {
                "Training Loss": train_loss,
            }

            if log_to_wandb:
                wandb.log(epoch_stats, step=epoch)

            if verbose:
                print("Epoch Stats: ", epoch_stats)

            train_losses.append(train_loss)

        loss = torch.tensor(train_losses)
        loss_mean = loss.mean().item()

        stats: Dict[str, float] = {
            "Mean Training Loss": loss_mean,
        }

        if log_to_wandb:
            for key, value in stats.items():
                wandb.run.summary[key] = value  # type: ignore

        if verbose:
            print("Training Stats: ", stats)

    def train_step(
        self, train_dataloder: Iterable, optimizer: torch.optim.Optimizer, **kwargs
    ) -> float:
        self.model.train()

        total_loss = 0
        for data in train_dataloder:
            optimizer.zero_grad()
            data = data.to(self.device)
            if hasattr(self.model, "train_step"):
                loss = self.model.train_step(
                    x=data.x, edge_index=data.edge_index, **kwargs
                )
            else:
                raise NotImplementedError
            loss.backward()
            total_loss += loss.item() * num_graphs(data)
            optimizer.step()

        return total_loss / len(train_dataloder.dataset)

    def val_step(self, val_dataloder: Iterable) -> float:
        raise NotImplementedError

    def test_step(self, test_dataloder: Iterable, test_metric: Optional[str]) -> float:
        raise NotImplementedError


def get_dataloader(dataset: Dataset, batch_size: int) -> Iterable:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
