import ml_collections
import pytest
import torch

from hivegraph.automodel import AutoModel


NUM_LAYERS: int = 5
HIDDEN_DIM: int = 32


@pytest.mark.nn
@pytest.mark.parametrize(
    "task, model_name",
    [
        ("classification", "GCN"),
        ("classification", "GIN"),
        ("transductive", "GRACE"),
    ],
)
def test_automodel(get_dataset, task: str, model_name: str) -> None:
    config = ml_collections.ConfigDict(
        {
            "model_name": model_name,
            "num_features": get_dataset.num_features,
            "num_layers": NUM_LAYERS,
            "hidden": HIDDEN_DIM,
        }
    )

    if task == "classification":
        config.update({"num_classes": get_dataset.num_classes})

    elif task == "transductive" and model_name == "GRACE":
        config.update(
            {
                "drop_edge_rate_1": 0.2,
                "drop_edge_rate_2": 0.4,
                "drop_feature_rate_1": 0.3,
                "drop_feature_rate_2": 0.4,
            }
        )

    # Assert Model is a torch.nn.Module
    model = AutoModel(task, config)
    assert isinstance(model, torch.nn.Module)

    # Assert Output shape
    if task == "classification":
        out = model(get_dataset[0])
        assert out.shape == (1, get_dataset.num_classes)
    elif task == "transductive":
        data = get_dataset[0]
        out = model(data.x, data.edge_index)
        assert out.shape == (len(data.x), HIDDEN_DIM)
