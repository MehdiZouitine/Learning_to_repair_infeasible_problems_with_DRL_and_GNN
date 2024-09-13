import torch
import torch_geometric
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from utils import lp_to_graph


def graph_to_pytorch_geometric_data(
    constraint_features, edge_indices, edge_features, variable_features
):
    # convert a batch of constraint features, edge indices, edge features, and variable features into a Batch of PyTorch Geometric Data objects
    # constraint_features: (batch_size, num_constraints, num_features)
    # edge_indices: (batch_size, 2, num_edges)
    # edge_features: (batch_size, num_edges, num_edge_features)
    # variable_features: (batch_size, num_variables, num_features)
    batch_size, num_constraints, num_features = constraint_features.shape
    batch = []
    for b in range(batch_size):
        batch.append(
            BipartiteNodeData(
                constraint_features=constraint_features[b],
                edge_indices=edge_indices[b],
                edge_features=edge_features[b],
                variable_features=variable_features[b],
            )
        )
    return torch_geometric.data.Batch.from_data_list(batch)


class GNNEncoder(torch.nn.Module):
    def __init__(self, cons_nfeats, edge_nfeats, var_nfeats, emb_size, n_layers=4):
        super().__init__()
        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size)
        # self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size)

        # self.conv_v_to_c2 = BipartiteGraphConvolution(emb_size=emb_size)
        # self.conv_c_to_v2 = BipartiteGraphConvolution(emb_size=emb_size)

        self.conv_v_to_c_layers = torch.nn.ModuleList(
            [BipartiteGraphConvolution(emb_size=emb_size) for _ in range(n_layers)]
        )
        self.conv_c_to_v_layers = torch.nn.ModuleList(
            [BipartiteGraphConvolution(emb_size=emb_size) for _ in range(n_layers)]
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        for conv_v_to_c, conv_c_to_v in zip(
            self.conv_v_to_c_layers, self.conv_c_to_v_layers
        ):
            # V -> C
            variable_features = conv_v_to_c(
                variable_features,
                reversed_edge_indices,
                edge_features,
                constraint_features,
            )
            # C -> V
            constraint_features = conv_c_to_v(
                constraint_features,
                edge_indices,
                edge_features,
                variable_features,
            )
            # Two half convolutions
        # constraint_features = self.conv_v_to_c(
        #     variable_features, reversed_edge_indices, edge_features, constraint_features
        # )
        # variable_features = self.conv_c_to_v(
        #     constraint_features, edge_indices, edge_features, variable_features
        # )

        # constraint_features = self.conv_v_to_c2(
        #     variable_features, reversed_edge_indices, edge_features, constraint_features
        # )
        # variable_features = self.conv_c_to_v2(
        #     constraint_features, edge_indices, edge_features, variable_features
        # )

        return constraint_features, variable_features


import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer("shift", torch.zeros(n_units) if shift else None)
        self.register_buffer("scale", torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert (
            self.n_units == 1 or input_.shape[-1] == self.n_units
        ), f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units

        delta = sample_avg - self.avg

        self.m2 = (
            self.var * self.count
            + sample_var * sample_count
            + delta**2 * self.count * sample_count / (self.count + sample_count)
        )

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
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
    def __init__(self, emb_size):
        super().__init__("add")

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(PreNormLayer(1, shift=False))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class BipartiteAgent(nn.Module):
    def __init__(
        self, cons_nfeats, edge_nfeats, var_nfeats, emb_size, n_layers=4, device="cpu"
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            cons_nfeats, edge_nfeats, var_nfeats, emb_size, n_layers
        )
        self.policy_head = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.ReLU(), nn.Linear(emb_size, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.ReLU(), nn.Linear(emb_size, 1)
        )
        self.device = device

    def get_value(self, constraint_features, edge_features, mask):
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
        constraint_features,
        edge_features,
        mask,
        action=None,
    ):
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
        cons_logits = cons_logits.masked_fill_(mask.bool(), float("-inf"))
        mass = Categorical(logits=cons_logits)
        if action is None:
            action = mass.sample()
        log_prob = mass.log_prob(action)
        entropy = mass.entropy()
        value = self.value_head(graph_embedding)

        return (
            action,
            log_prob,
            entropy,
            value,
        )

    def forward(self, constraint_features, edge_features, mask):
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
        cons_logits = cons_logits.masked_fill_(mask.bool(), float("-inf"))
        return cons_logits
