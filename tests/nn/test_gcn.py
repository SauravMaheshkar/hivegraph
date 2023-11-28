import pytest
import torch

from hivegraph.nn.gcn import GCN

NUM_LAYERS: int = 5
HIDDEN_DIM: int = 32


@pytest.mark.nn
@pytest.mark.parametrize(
    "use_jump, conv_variant, readout",
    [
        (False, "GCN", "mean"),
        (False, "GraphSAGE", "max"),
        (True, "ResGatedGraph", "sum"),
    ],
)
def test_gcn(get_dataset, use_jump: bool, conv_variant: str, readout: str) -> None:
    gnn = GCN(
        num_features=get_dataset.num_features,
        num_classes=get_dataset.num_classes,
        num_layers=NUM_LAYERS,
        hidden=HIDDEN_DIM,
        use_jump=use_jump,
        conv_variant=conv_variant,
        readout=readout,
    )
    # Assert Model is a torch.nn.Module
    assert isinstance(gnn, torch.nn.Module)

    # Assert Output shape
    out = gnn(get_dataset[0])
    assert out.shape == (1, get_dataset.num_classes)
