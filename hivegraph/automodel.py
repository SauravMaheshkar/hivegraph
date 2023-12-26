import ml_collections

from hivegraph.contrastive.grace import GRACE
from hivegraph.nn.gcn import GCN
from hivegraph.nn.gin import GIN


# TODO(@SauravMaheshkar): switch to using __dict__ syntax


def AutoModel(task: str, model_config: ml_collections.ConfigDict, **kwargs):
    if task == "classification":
        if model_config.model_name == "GIN":
            return GIN(**model_config, **kwargs)
        elif model_config.model_name == "GCN":
            return GCN(**model_config, **kwargs)
        else:
            raise ValueError(f"Model {model_config.model_name} not supported.")
    elif task == "contrastive":
        if model_config.model_name == "GRACE":
            return GRACE(**model_config, **kwargs)
        else:
            raise ValueError(f"Model {model_config.model_name} not supported.")
    else:
        raise ValueError(f"Task {task} not supported.")
