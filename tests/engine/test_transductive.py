import pytest
import torch

from hivegraph.engine.transductive import get_dataloader


@pytest.mark.io
def test_dataloder(get_transductive_dataset) -> None:
    dataloder = get_dataloader(get_transductive_dataset, 32)

    assert isinstance(dataloder, torch.utils.data.DataLoader)
