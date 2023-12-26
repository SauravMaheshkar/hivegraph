import ml_collections
import torch
from torch_geometric.nn import GCNConv


__all__ = ["get_config", "get_model_config"]


def get_config() -> ml_collections.ConfigDict:
    """Get Hyperparameter Configuration"""
    config = ml_collections.ConfigDict()

    config.task = "transductive"
    config.random_seed = 42  # Run for 7 and 3
    config.dataset = "Cora"
    config.num_epochs = 250
    config.batch_size = 256
    config.lr = 0.01
    config.verbose = True
    config.log_to_wandb = False

    config.wandb_project = "GRACE"
    config.wandb_entity = "graph-neural-networks"

    config.model = get_model_config()

    return config


def get_model_config() -> ml_collections.ConfigDict:
    """Get Model Hyperparameter Configuration"""
    model_config = ml_collections.ConfigDict()

    model_config.model_name = "GRACE"
    model_config.num_layers = 5
    model_config.hidden = 32
    model_config.projection_dim = 128
    model_config.tau = 0.5
    model_config.activation = torch.nn.functional.relu
    model_config.base_model = GCNConv
    model_config.drop_edge_rate_1 = 0.2
    model_config.drop_edge_rate_2 = 0.4
    model_config.drop_feature_rate_1 = 0.3
    model_config.drop_feature_rate_2 = 0.4

    return model_config
