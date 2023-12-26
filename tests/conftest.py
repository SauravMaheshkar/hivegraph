import pytest
from torch_geometric.data.dataset import Dataset

from hivegraph.io.autodataset import AutoDataset


@pytest.fixture
def get_dataset() -> Dataset:
    dataset = AutoDataset(dataset_name="MUTAG")
    return dataset.get_dataset()


@pytest.fixture
def get_transductive_dataset() -> Dataset:
    dataset = AutoDataset(dataset_name="Cora")
    return dataset.get_dataset()
