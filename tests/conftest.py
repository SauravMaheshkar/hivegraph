import pytest
from torch_geometric.data.dataset import Dataset


@pytest.fixture
def get_dataset() -> Dataset:
    from hivegraph.io.autodataset import AutoDataset

    dataset = AutoDataset(dataset_name="MUTAG")
    return dataset.get_dataset()
