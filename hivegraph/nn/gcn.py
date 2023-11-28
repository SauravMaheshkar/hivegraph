from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    JumpingKnowledge,
    ResGatedGraphConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

__all__: List[str] = ["GCN"]


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_layers: int,
        hidden: int,
        dropout: float = 0.5,
        use_jump: bool = False,
        conv_variant: str = "GCN",
        jump_mode: str = "cat",
        readout: str = "mean",
        model_name: str = "GCN",
    ) -> None:
        """
        Implementation of Graph Convolutional Network

        Args:
            num_features (int): Number of input features.
            num_classes (int): Number of output classes.
            num_layers (int): Number of GINConv layers.
            hidden (int): Number of hidden units.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
            use_jump (bool, optional): If True, use JumpingKnowledge to aggregate
                representations from all layers. Defaults to False.
            conv_variant (str, optional): Convolutional Layer to be used.
                Must be one of 'GCN' or 'ResGatedGraph'. Defaults to 'GCN'.
            jump_mode (str, optional): JumpingKnowledge aggregation mode.
                Must be one of 'cat', 'max', or 'lstm'. Defaults to 'cat'.
            readout (str, optional): Readout function. Must be one of 'mean',
                'max', or 'sum'. Defaults to 'mean'.

        References:
            * https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
        """
        super().__init__()
        self.use_jump = use_jump

        # Sanity Checks
        assert conv_variant in [
            "GCN",
            "GraphSAGE",
            "ResGatedGraph",
        ], "conv_variant must be on of 'GCN', or 'ResGatedGraph'"

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

        if conv_variant == "GCN":
            self.conv = GCNConv
        elif conv_variant == "GraphSAGE":
            self.conv = SAGEConv
        elif conv_variant == "ResGatedGraph":
            self.conv = ResGatedGraphConv

        self.initialization = self.conv(in_channels=num_features, out_channels=hidden)
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.conv_layers.append(self.conv(in_channels=hidden, out_channels=hidden))

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
        for conv_layer in self.conv_layers:
            conv_layer.reset_parameters()
        for layer in self.classification_head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.initialization(x, edge_index))
        if self.use_jump:
            xs = [x]
            for conv_layer in self.conv_layers:
                x = F.relu(conv_layer(x, edge_index))
                xs.append(x)
            x = self.jump(xs)
        else:
            for conv_layer in self.conv_layers:
                x = F.relu(conv_layer(x, edge_index))
        x = self.readout(x, batch)
        x = self.classification_head(x)
        return F.log_softmax(x, dim=-1)
