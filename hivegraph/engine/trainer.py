from typing import List

from hivegraph.engine.classification import ClassificationTrainer
from hivegraph.engine.transductive import TransductiveTrainer


__all__: List[str] = ["AutoTrainer"]


def AutoTrainer(task: str, **kwargs):
    """
    Helper function to instantiate a trainer based on the task.

    Args:
        task (str): Task to be performed by the trainer.
        **kwargs: Additional keyword arguments needed to instantiate the trainer.

    Returns:
        Instantiated trainer.

    Raises:
        ValueError: If the task is not supported.
    """
    if task == "classification":
        return ClassificationTrainer(**kwargs)
    elif task == "transductive":
        return TransductiveTrainer(**kwargs)
    else:
        raise ValueError(f"Task {task} not supported")
