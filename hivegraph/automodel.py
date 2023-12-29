from typing import List

from ml_collections import ConfigDict

from hivegraph.contrastive.grace import GRACE
from hivegraph.nn.gcn import GCN
from hivegraph.nn.gin import GIN


__all__: List[str] = ["AutoModel"]

# TODO(@SauravMaheshkar): switch to using __dict__ syntax


def AutoModel(task: str, model_config: ConfigDict, **kwargs):
    """
    Helper function to instantiate a model based on the task and model name.

    Args:
        task (str): Task to be performed by the model.
        model_config (ConfigDict): Model configuration.
        **kwargs: Additional keyword arguments needed to instantiate the model.

    Returns:
        torch.nn.Module: Instantiated model.

    Raises:
        ValueError: If the task or model name is not supported.
    """
    if task == "classification":
        if model_config.model_name == "GIN":
            return GIN(**model_config, **kwargs)
        elif model_config.model_name == "GCN":
            return GCN(**model_config, **kwargs)
        else:
            raise ValueError(f"Model {model_config.model_name} not supported.")
    elif task == "transductive":
        if model_config.model_name == "GRACE":
            return GRACE(**model_config, **kwargs)
        else:
            raise ValueError(f"Model {model_config.model_name} not supported.")
    else:
        raise ValueError(f"Task {task} not supported.")
