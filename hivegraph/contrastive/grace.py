from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge

from hivegraph.augmentations import drop_feature


__all__: List[str] = ["GRACE"]


class GRACE(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden: int,
        num_layers: int,
        drop_edge_rate_1: float,
        drop_edge_rate_2: float,
        drop_feature_rate_1: float,
        drop_feature_rate_2: float,
        activation: Callable = F.relu,
        base_model: torch.nn.Module = GCNConv,
        projection_dim: int = 128,
        tau: Optional[float] = 0.5,
        model_name: str = "GRACE",
        **kwargs,
    ) -> None:
        super(GRACE, self).__init__(**kwargs)
        self.encoder_module = GRACEEncoder(
            in_channels=num_features,
            out_channels=hidden,
            activation=activation,
            base_model=base_model,
            k=num_layers,
        )
        self.tau = tau

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, projection_dim),
            torch.nn.ELU(),
            torch.nn.Linear(projection_dim, projection_dim),
        )

        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2

    def reset_parameters(self) -> None:
        for conv_layer in self.encoder_module.conv:
            conv_layer.reset_parameters()
        for layer in self.projection_head:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass through the encoder module.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Representations from the encoder module.
        """
        return self.encoder_module(x, edge_index)

    def train_step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            float: Loss.
        """
        # Generate Graph Views

        ## Removing Edges (RE)
        edge_index_1 = dropout_edge(edge_index=edge_index, p=self.drop_edge_rate_1)[0]
        edge_index_2 = dropout_edge(edge_index=edge_index, p=self.drop_edge_rate_2)[0]

        ## Masking Node Features (MF)
        x_1 = drop_feature(x, drop_prob=self.drop_feature_rate_1)
        x_2 = drop_feature(x, drop_prob=self.drop_feature_rate_2)

        ## Generating views
        z1 = self.forward(x_1, edge_index_1)
        z2 = self.forward(x_2, edge_index_2)

        # Calculate Loss
        loss = self.loss(z1, z2, batch_size=0)

        return loss

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project the representations to a lower-dimensional space.

        This has been shown to enchance the expression power of the critic,
        For details refer to the section 3.2.1

        Ref: https://arxiv.org/pdf/2006.04131v2.pdf

        Args:
            z (torch.Tensor): Representations from the encoder module.

        Returns:
            torch.Tensor: Projected representations.
        """
        return self.projection_head(z)

    def compute_cosine_sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity between two sets of views.

        Args:
            z1 (torch.Tensor): First set of views.
            z2 (torch.Tensor): Second set of views.

        Returns:
            torch.Tensor: Cosine similarity between the two sets of views.
        """
        z1 = torch.nn.functional.normalize(z1)
        z2 = torch.nn.functional.normalize(z2)
        return torch.mm(z1, z2.T)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the "semi_loss" between two given views.

        Space Complexity: O(N^2)

        Args:
            z1 (torch.Tensor): First set of views.
            z2 (torch.Tensor): Second set of views.

        Returns:
            torch.Tensor: "semi_loss" between the two sets of views.
        """
        intraview_pairs = self.normalize_with_temp(self.compute_cosine_sim(z1, z1))
        interview_pairs = self.normalize_with_temp(self.compute_cosine_sim(z1, z2))

        return -torch.log(
            interview_pairs.diag()
            / (intraview_pairs.sum(1) + interview_pairs.sum(1) - interview_pairs.diag())
        )

    def batched_semi_loss(
        self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Calculate the "semi_loss" between a batch of views

        Space Complexity: O(BN)

        Args:
            z1 (torch.Tensor): First batch of views.
            z2 (torch.Tensor): Second batch of views.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: "semi_loss" between the two batches.
        """
        # Helper variables
        device: torch.device = z1.device
        num_nodes: int = z1.size(0)
        num_batches: int = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(device)
        losses: List[torch.Tensor] = []

        for i in range(num_batches):
            # Mask out other values not in the current batch
            mask: torch.Tensor = indices[i * batch_size : (i + 1) * batch_size]

            # Similar to self.semi_loss()
            intraview_pairs: torch.Tensor = self.normalize_with_temp(
                self.compute_cosine_sim(z1[mask], z1)
            )  # (batch_size, num_nodes)
            interview_pairs: torch.Tensor = self.normalize_with_temp(
                self.compute_cosine_sim(z1[mask], z2)
            )  # (batch_size, num_nodes)

            current_batch_loss: torch.Tensor = -torch.log(
                interview_pairs[:, i * batch_size : (i + 1) * batch_size].diag()
                / (
                    intraview_pairs.sum(1)
                    + interview_pairs.sum(1)
                    - intraview_pairs[:, i * batch_size : (i + 1) * batch_size].diag()
                )
            )

            losses.append(current_batch_loss)

        assert (
            len(losses) == num_batches
        ), "Number of losses must equal number of batches"

        return torch.cat(losses)

    def loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        mean: Optional[bool] = True,
        batch_size: int = 0,
    ) -> torch.Tensor:
        """
        Compute the overall loss for all positive pairs.

        Eqn(2) from the paper.

        Ref: https://arxiv.org/pdf/2006.04131v2.pdf

        Args:
            z1 (torch.Tensor): First set of views.
            z2 (torch.Tensor): Second set of views.
            mean (bool, optional): If True, return the mean loss. Defaults to True.
            batch_size (int, optional): Batch size. Defaults to 0.

        Returns:
            torch.Tensor: Overall loss.
        """
        # Generate views
        # one is used as the anchor
        # other forms the positive sample
        u: torch.Tensor = self.project(z1)
        v: torch.Tensor = self.project(z2)

        # As the two views are symmetric
        # the other loss is just calculated
        # using alternate parameters

        if batch_size == 0:
            l1: torch.Tensor = self.semi_loss(u, v)
            l2: torch.Tensor = self.semi_loss(v, u)
        else:
            l1: torch.Tensor = self.batched_semi_loss(u, v, batch_size)  # type: ignore[no-redef]
            l2: torch.Tensor = self.batched_semi_loss(v, u, batch_size)  # type: ignore[no-redef]

        loss: torch.Tensor = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()

        return loss

    def normalize_with_temp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given tensor with the temperature.

        Args:
            x (torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return torch.exp(x / self.tau)


class GRACEEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Callable = F.relu,
        base_model=GCNConv,
        k: int = 2,
    ) -> None:
        super(GRACEEncoder, self).__init__()
        self.base_model = base_model

        assert k >= 2, "k needs to be atleast 2"
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = torch.nn.ModuleList(self.conv)  # type: ignore[assignment]
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x
