from typing import List

from hivegraph.engine.classification import ClassificationTrainer
from hivegraph.engine.transductive import TransductiveTrainer


__all__: List[str] = ["AutoTrainer"]


def AutoTrainer(task: str, **kwargs):
    if task == "classification":
        return ClassificationTrainer(**kwargs)
    elif task == "transductive":
        return TransductiveTrainer(**kwargs)
    else:
        raise ValueError(f"Task {task} not supported")
