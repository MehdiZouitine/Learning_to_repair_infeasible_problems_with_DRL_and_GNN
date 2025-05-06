import torch
import torch.nn as nn
import torch_geometric
import numpy as np
from torch.distributions import Categorical
from typing import Optional, Tuple

from utils import lp_to_graph


class PreNormException(Exception):
    """Custom exception used for interrupting forward passes during pre-normalization stats collection."""
    pass


class PreNormLayer(nn.Module):
    """
    Applies affine transformation using precomputed shift and scale parameters.

    This is useful for pre-normalizing input features before training.
    """

    def __init__(self, n_units: int, shift: bool = True, scale: bool = True, name: Optional[str] = None):
        super().__init__()
        assert shift or scale
        self.register_buffer("shift", torch.zeros(n_units) if shift else None)
        self.register_buffer("scale", torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException()

        if self.shift is not None:
            input_ = input_ + self.shift
        if self.scale is not None:
            input_ = input_ * self.scale
        return input_

    def start_updates(self):
        """Start collecting statistics for normalization."""
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_: torch.Tensor):
        """Updates online mean and variance estimates."""
        assert input_.shape[-1] == self.n_units or self.n_units == 1
        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units

        delta = sample_avg - self.avg
        self.m2 = (
            self.var * self.count +
            sample_var * sample_count +
            delta ** 2 * self.count * sample_count / (self.count + sample_count)
        )
        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """Stops updates and freezes normalization parameters."""
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg
        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)
        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    Message passing layer for bipartite graphs (constraints ↔ variables).
    """

    def __init__(self, emb_size: int):
        super().__init__("add")
        self.feature_module_left = nn.Linear(emb_size, emb_size)
        self.feature_module_edge = nn.Linear(1, emb_size, bias=False)
        self.feature_module_right = nn.Linear(emb_size, emb_size, bias=False)

        self.feature_module_final = nn.Sequential(
            PreNormLayer(1, shift=False),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = nn.Sequential(PreNormLayer(1, shift=False))

        self.output_module = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(
        self,
        left_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        right_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform message passing and aggregate information into right_features.

        Returns:
            Updated right node features.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(
        self,
        node_features_i: torch.Tensor,
        node_features_j: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute messages passed from node i to node j.
        """
        return self.feature_module_final(
            self.feature_module_left(node_features_i) +
            self.feature_module_edge(edge_features) +
            self.feature_module_right(node_features_j)
        )


class GNNEncoder(nn.Module):
    """
    GNN Encoder for bipartite graphs of constraints and variables.

    Args:
        cons_nfeats: Number of constraint features.
        edge_nfeats: Number of edge features.
        var_nfeats: Number of variable features.
        emb_size: Size of the embeddings.
        n_layers: Number of GNN layers.
    """

    def __init__(self, cons_nfeats: int, edge_nfeats: int, var_nfeats: int,
                 emb_size: int, n_layers: int = 4):
        super().__init__()

        self.cons_embedding = nn.Sequential(
            PreNormLayer(cons_nfeats),
            nn.Linear(cons_nfeats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.edge_embedding = nn.Sequential(PreNormLayer(edge_nfeats))
        self.var_embedding = nn.Sequential(
            PreNormLayer(var_nfeats),
            nn.Linear(var_nfeats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )

        self.conv_v_to_c_layers = nn.ModuleList([
            BipartiteGraphConvolution(emb_size=emb_size) for _ in range(n_layers)
        ])
        self.conv_c_to_v_layers = nn.ModuleList([
            BipartiteGraphConvolution(emb_size=emb_size) for _ in range(n_layers)
        ])

    def forward(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform message passing for a given bipartite graph.

        Returns:
            Updated constraint and variable features.
        """
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        for conv_v_to_c, conv_c_to_v in zip(
            self.conv_v_to_c_layers, self.conv_c_to_v_layers
        ):
            variable_features = conv_v_to_c(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            constraint_features = conv_c_to_v(
                constraint_features, edge_indices, edge_features, variable_features
            )

        return constraint_features, variable_features


class BipartiteAgent(nn.Module):
    """
    Reinforcement learning agent using GNN encoder for bipartite graphs.
    """

    def __init__(
        self,
        cons_nfeats: int,
        edge_nfeats: int,
        var_nfeats: int,
        emb_size: int,
        n_layers: int = 4,
        device: str = "cpu"
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            cons_nfeats, edge_nfeats, var_nfeats, emb_size, n_layers
        )
        self.policy_head = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )
        self.device = device

    def get_value(
        self,
        constraint_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value estimates for a batch of observations.

        Returns:
            Value tensor of shape [batch_size, 1].
        """
        batch_size, n_const = mask.shape
        graph = lp_to_graph(constraint_features, edge_features, mask)

        constraint_features, _ = self.encoder(
            graph.constraint_features,
            graph.edge_index,
            graph.edge_attr,
            graph.variable_features,
        )
        constraint_features = constraint_features.view(batch_size, n_const, -1)
        graph_embedding = constraint_features.mean(dim=1)
        value = self.value_head(graph_embedding)
        return value

    def get_action_and_value(
        self,
        constraint_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute value estimates and log probabilities.

        Returns:
            action, log_prob, entropy, value
        """
        batch_size, n_const = mask.shape
        graph = lp_to_graph(constraint_features, edge_features, mask)
        constraint_features, _ = self.encoder(
            graph.constraint_features,
            graph.edge_index,
            graph.edge_attr,
            graph.variable_features,
        )

        constraint_features = constraint_features.view(batch_size, n_const, -1)
        graph_embedding = constraint_features.mean(dim=1)

        cons_logits = self.policy_head(constraint_features).squeeze(-1)
        cons_logits = cons_logits.masked_fill(mask.bool(), float("-inf"))

        mass = Categorical(logits=cons_logits)
        if action is None:
            action = mass.sample()

        log_prob = mass.log_prob(action)
        entropy = mass.entropy()
        value = self.value_head(graph_embedding)

        return action, log_prob, entropy, value

    def forward(
        self,
        constraint_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logits for all actions.

        Returns:
            Logits tensor of shape [batch_size, n_constraints].
        """
        batch_size, n_const = mask.shape
        graph = lp_to_graph(constraint_features, edge_features, mask)
        constraint_features, _ = self.encoder(
            graph.constraint_features,
            graph.edge_index,
            graph.edge_attr,
            graph.variable_features,
        )

        constraint_features = constraint_features.view(batch_size, n_const, -1)
        cons_logits = self.policy_head(constraint_features).squeeze(-1)
        cons_logits = cons_logits.masked_fill(mask.bool(), float("-inf"))
        return cons_logits
