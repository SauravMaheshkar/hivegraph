import ml_collections
import torch.nn as nn

__all__ = ["get_config", "get_model_config"]


def get_config() -> ml_collections.ConfigDict:
    """Get Hyperparameter Configuration"""
    config = ml_collections.ConfigDict()

    config.random_seed = 3  # Run for 7 and 3
    config.dataset = "MUTAG"
    config.criterion = (
        nn.functional.nll_loss
    )  # or nn.BCEWithLogitsLoss() for multi-class, nn.functional.nll_loss for binary
    config.num_folds = 10
    config.test_metric = "Accuracy"  # "AUROC" for multi-class, "Accuracy" for binary
    config.num_epochs = 250
    config.batch_size = 256
    config.lr = 0.01
    config.verbose = True
    config.log_to_wandb = False

    config.wandb_project = "GIN"
    config.wandb_entity = "graph-neural-networks"

    config.model = get_model_config()

    return config


def get_model_config() -> ml_collections.ConfigDict:
    """Get Model Hyperparameter Configuration"""
    model_config = ml_collections.ConfigDict()

    model_config.model_name = "gin"
    model_config.num_layers = 5
    model_config.hidden = 32
    model_config.use_eps = False
    model_config.use_jump = False
    model_config.dropout = 0.5
    model_config.batchnorm = "last"
    model_config.readout = "mean"

    return model_config
