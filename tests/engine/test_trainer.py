import ml_collections
import pytest
import torch

from hivegraph.automodel import AutoModel
from hivegraph.engine import AutoTrainer


NUM_LAYERS: int = 5
HIDDEN_DIM: int = 32


@pytest.mark.engine
@pytest.mark.parametrize(
    "task, model_name",
    [("classification", "GCN"), ("classification", "GIN"), ("transductive", "GRACE")],
)
def test_autotrainer(get_dataset, task: str, model_name: str) -> None:
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

    # Assert Model is a torch.nn.Module
    model = AutoModel(task, config)
    assert isinstance(model, torch.nn.Module)

    if task == "classification":
        trainer = AutoTrainer(
            task=task,
            model=model,
            dataset=get_dataset,
            device="cpu",
            criterion=torch.nn.functional.nll_loss,
            num_folds=2,
            random_state=42,
            test_metric="Accuracy",
        )
        assert trainer is not None
    elif task == "contrastive":
        trainer = AutoTrainer(
            task=task,
            model=model,
            dataset=get_dataset,
        )
        assert trainer is not None
