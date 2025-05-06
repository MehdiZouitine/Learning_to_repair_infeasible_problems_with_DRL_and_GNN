import numpy as np
from scipy.optimize import linprog
import torch
from torch_geometric.data import Data, Batch
import gymnasium as gym
import random

from pysat.solvers import Solver
from typing import List, Tuple, Optional

BASE_SEED = 10000


def create_parallel_env(base_env, **kwargs):
    """
    Creates a single environment instance.

    Args:
        base_env: The base Gym environment class.
        **kwargs: Additional keyword arguments to pass to the environment constructor.

    Returns:
        An instance of the environment.
    """
    return base_env(**kwargs)


def make_parallel_env(num_envs: int, base_env, **kwargs) -> gym.vector.AsyncVectorEnv:
    """
    Creates a vectorized environment with multiple instances running in parallel.

    Args:
        num_envs: Number of environment instances.
        base_env: The Gym environment class to instantiate.
        **kwargs: Additional keyword arguments for the environment constructor.

    Returns:
        A Gym AsyncVectorEnv containing the parallel environments.
    """
    envs = []
    for _ in range(num_envs):
        env = create_parallel_env(base_env, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        envs.append(lambda: env)
    return gym.vector.AsyncVectorEnv(envs)


def generate_random_lp(
    batch_size: int, n: int, m: int, weight: str = "const", seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a batch of random linear programming (LP) instances.

    Args:
        batch_size: Number of LP instances to generate.
        n: Number of variables.
        m: Number of constraints (inequalities).
        weight: "const" assigns weight 1 to all constraints, otherwise random weights.
        seed: Random seed for reproducibility.

    Returns:
        c: Objective coefficients of shape [batch_size, n].
        A_ub: Inequality matrix of shape [batch_size, m, n].
        b_ub: Right-hand side vector of shape [batch_size, m].
        constraint_weight: Weights for each constraint, shape [batch_size, m].
    """
    if seed:
        np.random.seed(seed)

    c = np.random.randint(-100, 101, (batch_size, n)) * 0
    A_ub = np.random.randint(-100, 101, (batch_size, m, n))
    b_ub = np.random.randint(-100, 101, (batch_size, m))

    if weight == "const":
        constraint_weight = np.ones_like(b_ub)
    else:
        constraint_weight = np.random.uniform(0, 1, (batch_size, m))

    return c, A_ub, b_ub, constraint_weight


def generate_random_infeasible_lp(
    batch_size: int, n: int, m: int, weight: str = "const", seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an infeasible LP by repeatedly generating random instances until infeasibility is found.

    Args:
        batch_size: Number of instances to generate (typically 1).
        n: Number of variables.
        m: Number of constraints.
        weight: Constraint weight type.
        seed: Random seed.

    Returns:
        Same as generate_random_lp: c, A_ub, b_ub, constraint_weight.
    """
    feasible = True
    k = 0
    while feasible:
        current_seed = k * BASE_SEED + seed if seed is not None else None
        c, A_ub, b_ub, constraint_weight = generate_random_lp(
            batch_size, n, m, weight, seed=current_seed
        )
        feasible = check_lp_feasibility(A_ub[0], b_ub[0])
        k += 1

    return c, A_ub, b_ub, constraint_weight


def lp_to_matrix(
    constraint_features: torch.Tensor,
    edge_features: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Converts LP features into a combined tensor for neural network input.

    Args:
        constraint_features: Tensor of shape [batch_size, n_cons, n_cons_features].
        edge_features: Tensor of shape [batch_size, n_cons, n_var, n_edge_features].
        mask: Tensor of shape [batch_size, n_cons] (unused here but often used in later processing).

    Returns:
        A tensor combining edge and constraint features of shape
        [batch_size, n_cons, n_var, n_edge_features + n_cons_features].
    """
    repeated_constraint_features = constraint_features.unsqueeze(2).repeat(
        1, 1, edge_features.size(2), 1
    )
    return torch.cat((edge_features, repeated_constraint_features), dim=-1)


class BipartiteNodeData(Data):
    """
    PyTorch Geometric data structure for bipartite graphs representing LPs or CNFs.

    Args:
        constraint_features: Constraint feature tensor.
        edge_indices: Edge index tensor [2, num_edges].
        edge_features: Edge feature tensor.
        variable_features: Variable feature tensor.
    """

    def __init__(
        self,
        constraint_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        variable_features: torch.Tensor,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        Increment function for batching multiple graphs (PyG requirement).
        """
        if key == "edge_index":
            return torch.tensor([
                self.constraint_features.size(0),
                self.variable_features.size(0)
            ])
        return super().__inc__(key, value, *args, **kwargs)


def lp_to_graph(
    constraint_features: torch.Tensor,
    edge_features: torch.Tensor,
    mask: torch.Tensor
) -> Batch:
    """
    Converts LP problem into a PyTorch Geometric bipartite graph batch.

    Args:
        constraint_features: Tensor [batch, n_cons, n_cons_features].
        edge_features: Tensor [batch, n_cons, n_var, n_edge_features].
        mask: Binary mask tensor [batch, n_cons].

    Returns:
        Batch object containing bipartite graphs for each batch element.
    """
    B, n_cons, n_var, _ = edge_features.shape
    device = edge_features.device

    variable_features = torch.zeros(B, n_var, 1, device=device)

    edge_indices = torch.stack([
        torch.arange(n_cons, device=device).repeat_interleave(n_var),
        torch.arange(n_var, device=device).repeat(n_cons)
    ])

    edge_features_flat = edge_features.flatten(start_dim=1, end_dim=-2)

    batch = []
    for b in range(B):
        batch.append(
            BipartiteNodeData(
                constraint_features=constraint_features[b],
                edge_indices=edge_indices,
                edge_features=edge_features_flat[b],
                variable_features=variable_features[b],
            )
        )
    return Batch.from_data_list(batch)


def check_lp_feasibility(A_ub: np.ndarray, b_ub: np.ndarray) -> bool:
    """
    Checks if the LP (minimize 0 subject to A_ub x <= b_ub) is feasible.

    Args:
        A_ub: Coefficient matrix [m, n].
        b_ub: Right-hand side vector [m].

    Returns:
        True if feasible, False otherwise.
    """
    res = linprog(
        np.zeros(A_ub.shape[1]),
        A_ub=A_ub,
        b_ub=b_ub,
        method="highs",
        bounds=(0, None)
    )
    return res.status != 2  # 2 = infeasible


def generate_random_cnf(
    num_clauses: int,
    num_variables: int,
    weight: str = "const",
    seed: Optional[int] = None,
    k_max: int = 3
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Generates a random CNF formula.

    Args:
        num_clauses: Number of clauses.
        num_variables: Number of variables.
        weight: "const" for equal weights, otherwise random weights.
        seed: Random seed.
        k_max: Maximum literals per clause.

    Returns:
        cnf: List of clauses (each clause is a list of ints).
        weights: Weight array for the clauses.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cnf = []
    for _ in range(num_clauses):
        k = k_max
        variables = random.sample(range(1, num_variables + 1), k)
        clause = [
            var if random.choice([True, False]) else -var for var in variables
        ]
        cnf.append(clause)

    if weight == "const":
        weights = np.ones(num_clauses)
    else:
        weights = np.random.uniform(0, 1, num_clauses)

    return cnf, weights


def check_sat_satisfiability(cnf_formula: List[List[int]]) -> bool:
    """
    Checks whether a CNF formula is satisfiable.

    Args:
        cnf_formula: List of clauses, each clause is a list of integers.

    Returns:
        True if satisfiable, False otherwise.
    """
    solver = Solver(name="g4")
    solver.append_formula(cnf_formula)
    result = solver.solve()
    solver.delete()
    return result


def generate_random_infeasible_cnf(
    num_clauses: int,
    num_variables: int,
    weight: str = "const",
    seed: Optional[int] = None,
    k_max: int = 3
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Generates a random CNF formula guaranteed to be infeasible.

    Args:
        num_clauses: Number of clauses.
        num_variables: Number of variables.
        weight: "const" for equal weights, otherwise random weights.
        seed: Random seed.
        k_max: Maximum literals per clause.

    Returns:
        cnf: Infeasible CNF formula.
        weights: Clause weights.
    """
    feasible = True
    k = 0
    while feasible:
        current_seed = k * BASE_SEED + seed if seed is not None else None
        cnf, weights = generate_random_cnf(
            num_clauses, num_variables, weight, seed=current_seed, k_max=k_max
        )
        feasible = check_sat_satisfiability(cnf)
        k += 1
    return cnf, weights


def cnf_to_matrix(cnf: List[List[int]], num_variables: int) -> np.ndarray:
    """
    Converts CNF formula to matrix form.

    Args:
        cnf: CNF formula as list of clauses.
        num_variables: Number of variables.

    Returns:
        Matrix [num_clauses, num_variables] with 1 (positive), -1 (negated), 0 (absent).
    """
    num_clauses = len(cnf)
    matrix = np.zeros((num_clauses, num_variables), dtype=int)

    for i, clause in enumerate(cnf):
        for literal in clause:
            idx = abs(literal) - 1
            matrix[i, idx] = 1 if literal > 0 else -1

    return matrix


if __name__ == "__main__":
    # Example test for infeasible CNF generation
    n = 15  # Number of variables
    m = 225  # Number of clauses
    num_instances = 1000
