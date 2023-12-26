from typing import Callable

import pytest
import torch
from torch_geometric.nn import GCNConv, GINConv

from hivegraph.contrastive import GRACE


NUM_LAYERS: int = 5
HIDDEN_DIM: int = 32


@pytest.mark.nn
@pytest.mark.parametrize(
    "activation, base_model, projection_dim",
    [
        (torch.nn.functional.relu, GCNConv, 128),
        (torch.nn.functional.elu, GINConv, 128),
    ],
)
def test_grace(
    get_dataset,
    activation: Callable,
    base_model: torch.nn.Module,
    projection_dim: int,
) -> None:
    grace = GRACE(
        num_features=get_dataset.num_features,
        num_layers=NUM_LAYERS,
        hidden=HIDDEN_DIM,
    )
    # Assert Model is a torch.nn.Module
    assert isinstance(grace, torch.nn.Module)

    # Assert Output shape
    data = get_dataset[0]
    out = grace(data.x, data.edge_index)
    assert out.shape == (len(data.x), 32)
