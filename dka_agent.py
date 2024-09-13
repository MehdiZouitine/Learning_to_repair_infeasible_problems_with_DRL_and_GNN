import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange


from torch_geometric.data import Data
from torch.distributions import Categorical

from utils import lp_to_matrix


class Normalization(nn.Module):
    """
    1D batch normalization for [*, C] input
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        size = x.size()
        return self.norm(x.reshape(-1, size[-1])).reshape(*size)


class Attention(nn.Module):
    """
    Multi-head attention

    Input:
      q: [batch_size, n_node, hidden_dim]
      k, v: q if None

    Output:
      att: [n_node, hidden_dim]
    """

    def __init__(
        self, q_hidden_dim, k_dim, v_dim, n_head, k_hidden_dim=None, v_hidden_dim=None
    ):
        super().__init__()
        self.q_hidden_dim = q_hidden_dim
        self.k_hidden_dim = k_hidden_dim if k_hidden_dim else q_hidden_dim
        self.v_hidden_dim = v_hidden_dim if v_hidden_dim else q_hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.proj_q = nn.Linear(q_hidden_dim, k_dim * n_head, bias=False)
        self.proj_k = nn.Linear(self.k_hidden_dim, k_dim * n_head, bias=False)
        self.proj_v = nn.Linear(self.v_hidden_dim, v_dim * n_head, bias=False)
        self.proj_output = nn.Linear(v_dim * n_head, self.v_hidden_dim, bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        if v is None:
            v = q

        bsz, n_node, hidden_dim = q.size()

        qs = torch.stack(
            torch.chunk(self.proj_q(q), self.n_head, dim=-1), dim=1
        )  # [batch_size, n_head, n_node, k_dim]
        ks = torch.stack(
            torch.chunk(self.proj_k(k), self.n_head, dim=-1), dim=1
        )  # [batch_size, n_head, n_node, k_dim]
        vs = torch.stack(
            torch.chunk(self.proj_v(v), self.n_head, dim=-1), dim=1
        )  # [batch_size, n_head, n_node, v_dim]

        normalizer = self.k_dim**0.5
        u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            u = u.masked_fill(mask.bool(), float("-inf"))
        att = torch.matmul(torch.softmax(u, dim=-1), vs)
        att = att.transpose(1, 2).reshape(bsz, n_node, self.v_dim * self.n_head)
        att = self.proj_output(att)
        return att


class DBALayer(nn.Module):
    def __init__(self, hidden_dim, ff_dim, k_dim, v_dim, n_head):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.attention_var = Attention(hidden_dim, k_dim, v_dim, n_head)
        self.attention_cons = Attention(hidden_dim, k_dim, v_dim, n_head)
        self.ff_var = nn.Sequential(
            *[
                nn.Linear(hidden_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, hidden_dim),
            ]
        )
        self.ff_cons = nn.Sequential(
            *[
                nn.Linear(hidden_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, hidden_dim),
            ]
        )

        self.bn1_var = Normalization(hidden_dim)
        self.bn2_var = Normalization(hidden_dim)
        self.bn1_cons = Normalization(hidden_dim)
        self.bn2_cons = Normalization(hidden_dim)

    def forward(self, x):
        B, K, N, H = x.size()

        x = rearrange(x, "b k n h -> (b k) n h", b=B, k=K, n=N, h=H)
        x = self.bn1_var(x + self.attention_var(x))
        x = self.bn2_var(x + self.ff_var(x))
        x = rearrange(x, "(b k) n h -> b k n h", b=B, k=K, n=N, h=H)
        x = rearrange(x, "b k n h -> (b n) k h", b=B, k=K, n=N, h=H)
        x = self.bn1_cons(x + self.attention_cons(x))
        x = self.bn2_cons(x + self.ff_cons(x))
        x = rearrange(x, "(b n) k h -> b k n h", b=B, k=K, n=N, h=H)
        return x


class DBAAgent(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, ff_dim, k_dim, v_dim, n_head, n_layers, device
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head
        self.n_layers = n_layers

        self.projector = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                DBALayer(hidden_dim, ff_dim, k_dim, v_dim, n_head)
                for _ in range(n_layers)
            ]
        )
        self.policy_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.device = device

    def get_value(self, constraint_features, edge_features, mask):
        x = lp_to_matrix(
            constraint_features=constraint_features,
            edge_features=edge_features,
            mask=mask,
        )
        x = self.projector(x)
        for layer in self.layers:
            x = layer(x)
        cons_embedding = x.mean(dim=2)
        graph_embedding = cons_embedding.mean(dim=1)
        return self.value_head(graph_embedding)

    def get_action_and_value(
        self, constraint_features, edge_features, mask, action=None
    ):
        x = lp_to_matrix(
            constraint_features=constraint_features,
            edge_features=edge_features,
            mask=mask,
        )
        x = self.projector(x)
        for layer in self.layers:
            x = layer(x)
        cons_embedding = x.mean(dim=2)
        graph_embedding = graph_embedding = cons_embedding.mean(dim=1)
        cons_logits = self.policy_head(cons_embedding).squeeze(-1)
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
        x = lp_to_matrix(
            constraint_features=constraint_features,
            edge_features=edge_features,
            mask=mask,
        )
        x = self.projector(x)
        for layer in self.layers:
            x = layer(x)
        cons_embedding = x.mean(dim=2)
        cons_logits = self.policy_head(cons_embedding).squeeze(-1)
        cons_logits = cons_logits.masked_fill_(mask.bool(), float("-inf"))
        return cons_logits


if __name__ == "__main__":
    from utils import make_parallel_env
    from env import MAXFSEnv
    import time
    from tqdm import tqdm

    agent = DBAAgent(
        input_dim=6,
        hidden_dim=128,
        ff_dim=512,
        k_dim=64,
        v_dim=64,
        n_head=3,
        n_layers=8,
        device="cuda:1",
    )
    agent.to("cuda:1")

    envs = make_parallel_env(16, MAXFSEnv, n_ineq_cons=200, n_vars=20)
    obs, info = envs.reset()
    x = torch.from_numpy(obs["matrix"]).float().to("cuda:1")
    mask = torch.from_numpy(obs["mask"]).bool().to("cuda:1")
    print(x.shape, mask.shape)
    start = time.time()
    for trial in tqdm(range(1000)):
        action, log_prob, entropy, value = agent.get_action_and_value(x, mask)
    print((time.time() - start) / 1000)
    envs.close()
