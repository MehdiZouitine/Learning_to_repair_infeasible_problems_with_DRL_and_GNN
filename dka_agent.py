import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch_geometric.data import Data
from torch.distributions import Categorical
from typing import Optional, Tuple

from utils import lp_to_matrix

# code implemented from https://link.springer.com/chapter/10.1007/978-3-031-60599-4_21

class Normalization(nn.Module):
    """
    1D batch normalization for tensors of shape [*, C].
    """

    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Size of the feature dimension to normalize.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.BatchNorm1d(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [*, feature_dim].

        Returns:
            Normalized tensor with the same shape.
        """
        size = x.size()
        return self.norm(x.reshape(-1, size[-1])).reshape(*size)


class Attention(nn.Module):
    """
    Multi-head self-attention module.
    """

    def __init__(
        self,
        q_hidden_dim: int,
        k_dim: int,
        v_dim: int,
        n_head: int,
        k_hidden_dim: Optional[int] = None,
        v_hidden_dim: Optional[int] = None
    ):
        """
        Args:
            q_hidden_dim: Dimension of the query vectors.
            k_dim: Dimension of key vectors per head.
            v_dim: Dimension of value vectors per head.
            n_head: Number of attention heads.
            k_hidden_dim: Key input dimension (defaults to q_hidden_dim).
            v_hidden_dim: Value input dimension (defaults to q_hidden_dim).
        """
        super().__init__()
        self.q_hidden_dim = q_hidden_dim
        self.k_hidden_dim = k_hidden_dim or q_hidden_dim
        self.v_hidden_dim = v_hidden_dim or q_hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.proj_q = nn.Linear(q_hidden_dim, k_dim * n_head, bias=False)
        self.proj_k = nn.Linear(self.k_hidden_dim, k_dim * n_head, bias=False)
        self.proj_v = nn.Linear(self.v_hidden_dim, v_dim * n_head, bias=False)
        self.proj_output = nn.Linear(v_dim * n_head, self.v_hidden_dim, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor [batch_size, n_node, hidden_dim].
            k: Key tensor [batch_size, n_node, hidden_dim] (defaults to q).
            v: Value tensor [batch_size, n_node, hidden_dim] (defaults to k).
            mask: Optional mask.

        Returns:
            Attention output tensor [batch_size, n_node, hidden_dim].
        """
        if k is None:
            k = q
        if v is None:
            v = k

        bsz, n_node, _ = q.size()

        qs = torch.stack(
            torch.chunk(self.proj_q(q), self.n_head, dim=-1), dim=1
        )
        ks = torch.stack(
            torch.chunk(self.proj_k(k), self.n_head, dim=-1), dim=1
        )
        vs = torch.stack(
            torch.chunk(self.proj_v(v), self.n_head, dim=-1), dim=1
        )

        normalizer = self.k_dim ** 0.5
        u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            u = u.masked_fill(mask.bool(), float("-inf"))

        att = torch.matmul(torch.softmax(u, dim=-1), vs)
        att = att.transpose(1, 2).reshape(bsz, n_node, self.v_dim * self.n_head)
        return self.proj_output(att)


class DBALayer(nn.Module):
    """
    Dual Bipartite Attention Layer (DBA).
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        k_dim: int,
        v_dim: int,
        n_head: int
    ):
        """
        Args:
            hidden_dim: Hidden embedding dimension.
            ff_dim: Feedforward layer dimension.
            k_dim: Attention key dimension per head.
            v_dim: Attention value dimension per head.
            n_head: Number of attention heads.
        """
        super().__init__()

        self.attention_var = Attention(hidden_dim, k_dim, v_dim, n_head)
        self.attention_cons = Attention(hidden_dim, k_dim, v_dim, n_head)

        self.ff_var = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.ff_cons = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.bn1_var = Normalization(hidden_dim)
        self.bn2_var = Normalization(hidden_dim)
        self.bn1_cons = Normalization(hidden_dim)
        self.bn2_cons = Normalization(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch_size, n_cons, n_var, hidden_dim].

        Returns:
            Updated tensor of the same shape.
        """
        B, K, N, H = x.size()

        x = rearrange(x, "b k n h -> (b k) n h")
        x = self.bn1_var(x + self.attention_var(x))
        x = self.bn2_var(x + self.ff_var(x))
        x = rearrange(x, "(b k) n h -> b k n h", b=B, k=K, n=N)

        x = rearrange(x, "b k n h -> (b n) k h")
        x = self.bn1_cons(x + self.attention_cons(x))
        x = self.bn2_cons(x + self.ff_cons(x))
        x = rearrange(x, "(b n) k h -> b k n h", b=B, k=K, n=N)

        return x


class DBAAgent(nn.Module):
    """
    Deep Reinforcement Learning agent using DBA layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        ff_dim: int,
        k_dim: int,
        v_dim: int,
        n_head: int,
        n_layers: int,
        device: str
    ):
        """
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Embedding dimension.
            ff_dim: Feedforward layer dimension.
            k_dim: Attention key dimension.
            v_dim: Attention value dimension.
            n_head: Number of attention heads.
            n_layers: Number of DBA layers.
            device: Device ("cpu" or "cuda").
        """
        super().__init__()
        self.device = device

        self.projector = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            DBALayer(hidden_dim, ff_dim, k_dim, v_dim, n_head)
            for _ in range(n_layers)
        ])

        self.policy_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_value(
        self,
        constraint_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value estimate.

        Args:
            constraint_features: Constraint features [batch_size, n_cons, feat_dim].
            edge_features: Edge features [batch_size, n_cons, n_var, edge_feat_dim].
            mask: Mask tensor [batch_size, n_cons].

        Returns:
            Value tensor [batch_size, 1].
        """
        x = lp_to_matrix(constraint_features, edge_features, mask)
        x = self.projector(x)
        for layer in self.layers:
            x = layer(x)
        cons_embedding = x.mean(dim=2)
        graph_embedding = cons_embedding.mean(dim=1)
        return self.value_head(graph_embedding)

    def get_action_and_value(
        self,
        constraint_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute policy action, log probability, entropy, and value.

        Args:
            constraint_features: Constraint features.
            edge_features: Edge features.
            mask: Mask.
            action: Optional pre-specified action.

        Returns:
            action: Selected action indices.
            log_prob: Log probabilities.
            entropy: Entropy values.
            value: Value estimates.
        """
        x = lp_to_matrix(constraint_features, edge_features, mask)
        x = self.projector(x)
        for layer in self.layers:
            x = layer(x)

        cons_embedding = x.mean(dim=2)
        graph_embedding = cons_embedding.mean(dim=1)
        cons_logits = self.policy_head(cons_embedding).squeeze(-1)
        cons_logits = cons_logits.masked_fill(mask.bool(), float("-inf"))

        dist = Categorical(logits=cons_logits)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(graph_embedding)

        return action, log_prob, entropy, value

    def forward(
        self,
        constraint_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action logits for the given state.

        Args:
            constraint_features: Constraint features.
            edge_features: Edge features.
            mask: Mask.

        Returns:
            Logits for actions.
        """
        x = lp_to_matrix(constraint_features, edge_features, mask)
        x = self.projector(x)
        for layer in self.layers:
            x = layer(x)

        cons_embedding = x.mean(dim=2)
        cons_logits = self.policy_head(cons_embedding).squeeze(-1)
        cons_logits = cons_logits.masked_fill(mask.bool(), float("-inf"))
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
        device="cuda:1"
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
    print(f"Average time per step: {(time.time() - start) / 1000:.6f} seconds")
    envs.close()
