from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from rich import print
from rich.progress import track
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

from hivegraph.io.autodataset import (
    BINARY_CLASSIFICATION_DATASTES,
    MULTI_CLASSIFICATION_DATASETS,
    SUPPORTED_DATASETS,
)

__all__: List[str] = ["Trainer"]


class Trainer:
    """
    Custom Trainer Class
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        criterion: Callable = F.nll_loss,
        device: str = "cpu",
        num_folds: int = 10,
        random_state: int = 42,
        test_metric: Optional[str] = None,
    ) -> None:
        """
        Args:
            model(torch.nn.Module): Model
            dataset(Dataset): PyTorch Geometric Dataset
            device(str): Device to run the model on, defaults to "cpu"
            num_folds(int): Number of folds for K-Fold Cross Validation, defaults to 10
            random_state(int): Random Seed, defaults to 42
            test_metric(str): Test Metric either "Accuracy" or "AUROC", defaults to None
        """
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.device = device
        self.num_folds = num_folds
        self.random_state = random_state
        self.test_metric = test_metric
        if self.test_metric is None:
            if self.dataset.name in BINARY_CLASSIFICATION_DATASTES:
                self.test_metric = "Accuracy"
            elif self.dataset.name in map(str.lower, MULTI_CLASSIFICATION_DATASETS):
                self.test_metric = "AUROC"
            else:
                raise ValueError(
                    f"Dataset not supported yet, as of now only {SUPPORTED_DATASETS} are supported"  # noqa: E501
                )

    def fit(
        self,
        epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        verbose: bool = True,
        log_to_wandb: bool = False,
    ) -> None:
        """
        Trains the model on the given dataset

        Args:
            epochs(int): Number of epochs
            batch_size(int): batch size
            optimizer(torch.optim.Optimizer): Optimizer
            verbose(bool): Whether to print stats, defaults to True
            log_to_wandb(bool): Whether to log stats to wandb, defaults to False
        """
        val_losses: List[float] = []
        accs: List[float] = []

        for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*get_kfold_indices(self.dataset, self.num_folds))
        ):
            train_dataset = self.dataset[train_idx]
            test_dataset = self.dataset[test_idx]
            val_dataset = self.dataset[val_idx]

            train_dataloder, val_dataloader, test_dataloader = get_dataloader(
                train_dataset, val_dataset, test_dataset, batch_size
            )

            self.model.to(self.device).reset_parameters()

            for epoch in track(range(1, epochs + 1), description=f"Fold {fold + 1}"):
                train_loss: float = self.train_step(
                    train_dataloder=train_dataloder, optimizer=optimizer
                )
                val_loss: float = self.val_step(val_dataloader=val_dataloader)
                test_acc: float = self.test_step(
                    test_dataloader=test_dataloader, test_metric=self.test_metric
                )

                epoch_stats: Dict[str, float] = {
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                    f"Test {self.test_metric}": test_acc,
                }

                if log_to_wandb:
                    wandb.log(epoch_stats, step=epoch)

                if verbose:
                    print("Epoch Stats: ", epoch_stats)

                val_losses.append(val_loss)
                accs.append(test_acc)

        loss, acc = torch.tensor(val_losses), torch.tensor(accs)
        loss, acc = loss.view(self.num_folds, epochs), acc.view(self.num_folds, epochs)
        loss, argmin = loss.min(dim=1)

        acc = acc[torch.arange(self.num_folds, dtype=torch.long), argmin]

        loss_mean = loss.mean().item()
        acc_mean = acc.mean().item()
        acc_std = acc.std().item()

        stats: Dict[str, Union[int, float]] = {
            "Final Validation Loss": loss_mean,
            f"Final Test {self.test_metric}": acc_mean,
            f"Final Test {self.test_metric} Std": acc_std,
        }

        if log_to_wandb:
            for key, value in stats.items():
                wandb.run.summary[key] = value  # type: ignore

        if verbose:
            print("Fold Stats: ", stats)

    def train_step(
        self, train_dataloder: Iterable, optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Performs a single training step

        Args:
            train_dataloder(Iterable): Training Dataloader
            optimizer(torch.optim.Optimizer): Optimizer

        Returns:
            float: Training Loss
        """
        self.model.train()

        total_loss = 0
        for data in train_dataloder:
            optimizer.zero_grad()
            data = data.to(self.device)
            out = self.model(data)
            if self.dataset.name in BINARY_CLASSIFICATION_DATASTES:
                loss = self.criterion(out, data.y.view(-1))
            elif self.dataset.name in map(str.lower, MULTI_CLASSIFICATION_DATASETS):
                loss = self.criterion(out, data.y)
            loss.backward()
            total_loss += loss.item() * num_graphs(data)
            optimizer.step()
        return total_loss / len(train_dataloder.dataset)

    def val_step(self, val_dataloader: Iterable) -> float:
        """
        Calculates the loss on the validation set

        Args:
            val_dataloader(Iterable): Validation Dataloader

        Returns:
            float: Validation Loss
        """
        self.model.eval()

        loss = 0
        for data in val_dataloader:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data)
            loss += self.criterion(out, data.y).item()
        return loss / len(val_dataloader.dataset)

    def test_step(
        self, test_dataloader: Iterable, test_metric: Optional[str]
    ) -> float:  # type: ignore
        """
        Calculates the accuracy on the test set

        Args:
            test_dataloader(Iterable): Test Dataloader
            test_metric(str): Test Metric, either "Accuracy" or "AUROC"

        Returns:
            float: Test Accuracy
        """
        self.model.eval()

        accs: List[float] = []

        for data in test_dataloader:
            data = data.to(self.device)
            if self.dataset.name in BINARY_CLASSIFICATION_DATASTES:
                if test_metric == "Accuracy":
                    accs.append(binary_accuracy(self.model, data).item())
                else:
                    raise ValueError(
                        f"Test metric {test_metric}, not supported for binary classification"  # noqa: E501
                    )
            elif self.dataset.name in map(str.lower, MULTI_CLASSIFICATION_DATASETS):
                if test_metric == "AUROC":
                    accs.append(multilabel_auroc(self.model, data).item())
                elif test_metric == "Average Precision":
                    accs.append(multilabel_average_precision(self.model, data).item())
                else:
                    raise ValueError(
                        f"Test metric {test_metric}, not supported for multilabel classification"  # noqa: E501
                    )

        return sum(accs) / len(accs)


def get_dataloader(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
) -> Tuple[Iterable, Iterable, Iterable]:
    """
    Utility Function to get dataloaders for train, val and test

    Args:
        train_dataset(Dataset): Training Dataset
        val_dataset(Dataset): Validation Dataset
        test_dataset(Dataset): Test Dataset
        batch_size(int): Batch Size

    Returns:
        A Tuple of 3 dataloaders for train, val and test, either of type
        torch_geometric.loader.DataLoader or torch_geometric.loader.DenseDataLoader
    """
    if "adj" in train_dataset[0]:
        train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_kfold_indices(
    dataset: Dataset, num_folds: int = 10, random_state: int = 42
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Utility Function to get indices for k-fold cross validation

    Args:
        dataset(Dataset): Dataset to split
        num_folds(int): Number of folds, defaults to 10
        random_state(int): Random seed, defaults to 42

    Returns:
        A Tuple of 3 lists of torch tensors containing mask indices for splits
    """
    if dataset.name in BINARY_CLASSIFICATION_DATASTES:
        skf = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state
        )
    elif dataset.name in map(str.lower, MULTI_CLASSIFICATION_DATASETS):
        skf = MultilabelStratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state
        )
    else:
        raise ValueError("Dataset not supported")

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset._data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(num_folds)]

    for i in range(num_folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


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


def binary_accuracy(model: torch.nn.Module, data: Data) -> torch.Tensor:
    """
    Calculate the accuracy on the test set for binary classification

    Args:
        model(torch.nn.Module): Model
        data(Data): Data object

    Returns:
        (torch.Tensor): Test Accuracy
    """
    with torch.no_grad():
        pred = model(data).max(1)[1]
        return torchmetrics.functional.classification.binary_accuracy(
            preds=pred, target=data.y
        )


def multilabel_auroc(
    model: torch.nn.Module,
    data: Data,
) -> torch.Tensor:
    """
    Calculate the AUROC Score on the test set for multilabel classification

    Args:
        model(torch.nn.Module): Model
        data(Data): Data object

    Returns:
        (torch.Tensor): Test AUROC Score
    """
    with torch.no_grad():
        predictions = model(data)
        return torchmetrics.functional.classification.multilabel_auroc(
            preds=predictions,
            target=(data.y).type(torch.int32),
            num_labels=data.y.size(1),
        )


def multilabel_average_precision(
    model: torch.nn.Module,
    data: Data,
) -> torch.Tensor:
    """
    Calculate the Average Precision Score on the test set for multilabel classification

    Args:
        model(torch.nn.Module): Model
        data(Data): Data object

    Returns:
        (torch.Tensor): Test Average Precision Score
    """
    with torch.no_grad():
        predictions = model(data)
        return torchmetrics.functional.classification.multilabel_average_precision(
            preds=predictions,
            target=(data.y).type(torch.int32),
            num_labels=data.y.size(1),
        )
