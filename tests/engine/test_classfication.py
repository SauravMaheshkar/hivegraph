import pytest
import torch

from hivegraph.engine.classification import get_dataloader, get_kfold_indices


@pytest.mark.engine
@pytest.mark.parametrize("num_folds", [5])
def test_get_kfold_indices(get_dataset, num_folds: int) -> None:
    train_indices, test_indices, val_indices = get_kfold_indices(get_dataset, num_folds)
    assert len(train_indices) == num_folds
    assert isinstance(train_indices, list)
    assert isinstance(train_indices[0], torch.Tensor)


@pytest.mark.engine
def test_get_dataloader(get_dataset) -> None:
    train_loader, val_loader, test_loader = get_dataloader(
        get_dataset, get_dataset, get_dataset, 32
    )
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
