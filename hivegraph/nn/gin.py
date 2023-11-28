from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GINConv,
    JumpingKnowledge,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

__all__: List[str] = ["GIN"]


class GIN(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_layers: int,
        hidden: int,
        use_eps: bool = False,
        use_jump: bool = False,
        jump_mode: str = "cat",
        dropout: float = 0.5,
        batchnorm: str = "sequential",
        readout: str = "mean",
        model_name: str = "GIN",
    ) -> None:
        """
        Implementation of Graph Isomorphism Networks (GIN) from
        “How Powerful are Graph Neural Networks?” <https://arxiv.org/abs/1810.00826>
        by Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka.

        Args:
            num_features (int): Number of input features.
            num_classes (int): Number of output classes.
            num_layers (int): Number of GINConv layers.
            hidden (int): Number of hidden units.
            use_eps (bool, optional): If True, epsilon is a learnable parameter.
                Defaults to False.
            use_jump (bool, optional): If True, use JumpingKnowledge to aggregate
                representations from all layers. Defaults to False.
            jump_mode (str, optional): JumpingKnowledge aggregation mode.
                Must be one of 'cat', 'max', or 'lstm'. Defaults to 'cat'.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
            batchnorm (str, optional): Batchnorm mode. Must be one of 'first',
                'last', or 'sequential'. Defaults to 'last'.
            readout (str, optional): Readout function. Must be one of 'mean',
                'max', or 'sum'. Defaults to 'mean'.

        References:
            * https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gin.py
        """
        super().__init__()
        self.use_jump = use_jump

        # Sanity Checks
        assert batchnorm in [
            "first",
            "last",
            "sequential",
        ], "batchnorm must be one of 'first', 'last', or 'sequential'"

        assert readout in [
            "mean",
            "max",
            "sum",
        ], "readout must be one of 'mean', 'max', or 'sum'"

        assert jump_mode in [
            "cat",
            "max",
            "lstm",
        ], "jump_mode must be one of 'cat', 'max', or 'lstm'"

        self.initialization = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden)
                if batchnorm in ["first", "sequential"]
                else nn.Identity(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden)
                if batchnorm in ["last", "sequential"]
                else nn.Identity(),
            ),
            train_eps=use_eps,
        )
        self.mp_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.mp_layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden)
                        if batchnorm in ["first", "sequential"]
                        else nn.Identity(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden)
                        if batchnorm in ["last", "sequential"]
                        else nn.Identity(),
                    ),
                    train_eps=use_eps,
                )
            )

        if self.use_jump:
            self.jump = JumpingKnowledge(mode=jump_mode)

        # Readout Function
        if readout == "mean":
            self.readout = global_mean_pool
        elif readout == "max":
            self.readout = global_max_pool
        elif readout == "sum":
            self.readout = global_add_pool

        self.classification_head = nn.Sequential(
            nn.Linear(hidden * num_layers, hidden)
            if (self.use_jump and jump_mode == "cat")
            else nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def reset_parameters(self) -> None:
        self.initialization.reset_parameters()
        for mp_layer in self.mp_layers:
            mp_layer.reset_parameters()
        for layer in self.classification_head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.initialization(x, edge_index)
        if self.use_jump:
            xs = [x]
            for mp_layer in self.mp_layers:
                x = mp_layer(x, edge_index)
                xs.append(x)
            x = self.jump(xs)
        else:
            for mp_layer in self.mp_layers:
                x = mp_layer(x, edge_index)
        x = self.readout(x, batch)
        x = self.classification_head(x)
        return F.log_softmax(x, dim=-1)
