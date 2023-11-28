import pytest
import torch

from hivegraph.nn.gin import GIN

NUM_LAYERS: int = 5
HIDDEN_DIM: int = 32


@pytest.mark.nn
@pytest.mark.parametrize(
    "batchnorm, num_batchnorm, readout, use_jump",
    [
        ("first", NUM_LAYERS, "mean", False),
        ("last", NUM_LAYERS, "max", False),
        ("sequential", 2 * NUM_LAYERS, "sum", True),
    ],
)
def test_gin(
    get_dataset, batchnorm: str, num_batchnorm: int, readout: str, use_jump: bool
) -> None:
    gnn = GIN(
        num_features=get_dataset.num_features,
        num_classes=get_dataset.num_classes,
        num_layers=NUM_LAYERS,
        hidden=HIDDEN_DIM,
        use_jump=use_jump,
        batchnorm=batchnorm,
        readout=readout,
    )
    # Assert Model is a torch.nn.Module
    assert isinstance(gnn, torch.nn.Module)

    # Assert Model has the correct number of batchnorm layers
    counter = 0
    for module in gnn.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            counter += 1

    assert counter == num_batchnorm

    # Assert Output shape
    out = gnn(get_dataset[0])
    assert out.shape == (1, get_dataset.num_classes)
