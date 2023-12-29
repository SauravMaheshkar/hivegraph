# mypy: disable-error-code="attr-defined,arg-type"
from typing import List

import torch
import torch_geometric.transforms as T
from torch_geometric.data.dataset import Dataset
from torch_geometric.datasets import CitationFull, LRGBDataset, Planetoid, TUDataset
from torch_geometric.utils import degree

from hivegraph.io import LRGB_DATASETS, TRANSDUCTIVE_DATASETS, TU_DATASETS
from hivegraph.io.transforms import NormalizedDegree


__all__: List[str] = ["AutoDataset"]


BINARY_CLASSIFICATION_DATASTES: List[str] = TU_DATASETS

MULTI_CLASSIFICATION_DATASETS: List[str] = LRGB_DATASETS

CLASSIFICATION_DATASETS: List[str] = (
    BINARY_CLASSIFICATION_DATASTES + MULTI_CLASSIFICATION_DATASETS
)

SUPPORTED_DATASETS: List[str] = (
    BINARY_CLASSIFICATION_DATASTES
    + MULTI_CLASSIFICATION_DATASETS
    + TRANSDUCTIVE_DATASETS
)


class AutoDataset:
    """
    Helper class to instantiate a dataset based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset.
        path (str, optional): Path to the dataset. Defaults to "artifacts/data/".
        sparse (bool, optional): If True, the dataset is returned as a sparse
            dataset. Defaults to True.
        cleaned (bool, optional): If True, the dataset is returned as a cleaned
            dataset. Defaults to False.

    References:

    * https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/datasets.py

    Raises:
        AssertionError: If the dataset name is not supported.
    """

    def __init__(
        self,
        dataset_name: str,
        path: str = "artifacts/data/",
        sparse: bool = True,
        cleaned: bool = False,
    ) -> None:
        assert dataset_name in SUPPORTED_DATASETS, (
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are {SUPPORTED_DATASETS}."
        )
        self.dataset_name = dataset_name
        self.path = path
        self.cleaned = cleaned
        self.sparse = sparse
        self.dataset = None

    def get_dataset(self) -> Dataset:
        """
        Fetches the dataset

        Returns:
            Dataset: PyG dataset.
        """
        if self.dataset is None:
            self.dataset = self.process()
        return self.dataset

    def download(self) -> Dataset:
        """
        Downloads the dataset

        Returns:
            Dataset: PyG dataset.

        Raises:
            NotImplementedError: If the dataset is not implemented.
        """
        if self.dataset_name in TU_DATASETS:
            self.dataset = TUDataset(
                root=self.path, name=self.dataset_name, cleaned=self.cleaned
            )
            self.dataset._data.edge_attr = None
        elif self.dataset_name in LRGB_DATASETS:
            self.dataset = LRGBDataset(root=self.path, name=self.dataset_name)
        elif self.dataset_name in TRANSDUCTIVE_DATASETS:
            self.dataset = (CitationFull if self.dataset_name == "DBLP" else Planetoid)(
                root=self.path, name=self.dataset_name, transform=T.NormalizeFeatures()
            )
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")

        return self.dataset

    def process(self) -> Dataset:
        """
        Processes the dataset

        Returns:
            Dataset: PyG dataset.

        Raises:
            NotImplementedError: If the dataset is not implemented.
        """
        self.dataset = self.download()

        if self.dataset_name in CLASSIFICATION_DATASETS:
            self.process_classification_dataset()
        elif self.dataset_name in TRANSDUCTIVE_DATASETS:
            self.process_transductive_dataset()
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")

        return self.dataset

    def process_classification_dataset(self) -> Dataset:
        """
        Processes the classification dataset

        Returns:
            Dataset: PyG dataset.
        """
        if self.dataset._data.x is None:
            max_degree = 0
            degs = []
            for data in self.dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                self.dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                self.dataset.transform = NormalizedDegree(mean, std)

        if not self.sparse:
            num_nodes = max_num_nodes = 0
            for data in self.dataset:
                num_nodes += data.num_nodes
                max_num_nodes = max(data.num_nodes, max_num_nodes)

            # Filter out a few really large graphs in order to apply DiffPool.
            if (
                self.dataset_name == "REDDIT-BINARY"
                or self.dataset_name in MULTI_CLASSIFICATION_DATASETS
            ):
                num_nodes = min(int(num_nodes / len(self.dataset) * 1.5), max_num_nodes)
            else:
                num_nodes = min(int(num_nodes / len(self.dataset) * 5), max_num_nodes)

            indices = []
            for i, data in enumerate(self.dataset):
                if data.num_nodes <= num_nodes:
                    indices.append(i)
            self.dataset = self.dataset.copy(torch.tensor(indices))

            if self.dataset.transform is None:
                self.dataset.transform = T.ToDense(num_nodes)
            else:
                self.dataset.transform = T.Compose(
                    [self.dataset.transform, T.ToDense(num_nodes)]
                )

    def process_transductive_dataset(self) -> Dataset:
        """
        Processes the transductive dataset

        Returns:
            Dataset: PyG dataset.
        """
        pass

    def __repr__(self) -> str:
        """
        String representation of the dataset.

        Returns:
            str: String representation of the dataset.
        """
        if self.dataset is None:
            return f"AutoDataset({self.dataset_name})"
        else:
            return f"AutoDataset({self.dataset_name}, {len(self.get_dataset())})"
