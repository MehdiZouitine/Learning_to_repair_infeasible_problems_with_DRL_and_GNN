import numpy as np
from scipy.optimize import linprog
import torch

from torch_geometric.data import Data, Batch
import gymnasium as gym
import random

from pysat.solvers import Solver
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

BASE_SEED = 10000


def create_parallel_env(base_env, **kwargs):
    return base_env(**kwargs)


def make_parallel_env(num_envs, base_env, **kwargs):
    envs = []
    for _ in range(num_envs):
        env = create_parallel_env(base_env, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        envs.append(lambda: env)
    vectorized_env = gym.vector.AsyncVectorEnv(envs)
    return vectorized_env


def generate_random_lp(batch_size, n, m, weight="const", seed=None):
    """
    Generates a batch of random linear programming problems that are likely infeasible.

    Parameters:
    - batch_size: Number of LP instances to generate.
    - n: Number of variables (dimension of the space).
    - m: Number of inequalities (constraints).

    Returns:
    - c: Objective function coefficients (batch_size, n).
    - A_ub: Inequality constraints coefficients (batch_size, m, n).
    - b_ub: Right-hand side of inequality constraints (batch_size, m).
    """
    # Random integer coefficients between -100 and 100
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


def generate_random_infeasible_lp(batch_size, n, m, weight="const", seed=None):
    feasible = True
    k = 0
    while feasible:
        if seed is not None:
            current_seed = k * BASE_SEED + seed
        else:
            current_seed = seed
        (c, A_ub, b_ub, constraint_weight) = generate_random_lp(
            batch_size, n, m, weight, seed=current_seed
        )
        feasible = check_lp_feasibility(A_ub[0], b_ub[0])
        k += 1
    return c, A_ub, b_ub, constraint_weight


def lp_to_matrix(constraint_features, edge_features, mask):
    # constraint_features: shape (batch_size, n_cons, n_cons_features)
    # edge_features: shape (batch_size, n_cons, n_var, n_edge_features)
    # mask: shape (batch_size, n_cons)

    # Repeat constraint_features along the third axis (n_var)
    repeated_constraint_features = constraint_features.unsqueeze(2).repeat(
        1, 1, edge_features.size(2), 1
    )

    # Concatenate edge_features, repeated_constraint_features, and repeated_mask
    matrix = torch.cat(
        (
            edge_features,
            repeated_constraint_features,
        ),
        dim=-1,
    )

    return matrix


class BipartiteNodeData(Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def lp_to_graph(constraint_features, edge_features, mask):
    # Given the coefficients of the LP, we can convert it to a bipartite graph
    # Return the constraint features, edge indices, edge features, and variable features
    # Use PyTorch tensors to represent the LP coefficients

    B, n_cons, n_var, n_edge_features = edge_features.shape
    device = edge_features.device  # Get the device from edge_features
    num_constraints, num_variables = n_cons, n_var

    # Create variable features tensor
    variable_features = torch.zeros(B, num_variables, 1, device=device)

    # Create edge indices tensor using torch
    edge_indices = torch.stack(
        [
            torch.arange(num_constraints, device=device).repeat_interleave(
                num_variables
            ),
            torch.arange(num_variables, device=device).repeat(num_constraints),
        ]
    )

    # Reshape edge features tensor
    edge_features = edge_features.flatten(start_dim=1, end_dim=-2)
    batch = []
    for b in range(B):
        batch.append(
            BipartiteNodeData(
                constraint_features=constraint_features[b],
                edge_indices=edge_indices,
                edge_features=edge_features[b],
                variable_features=variable_features[b],
            )
        )
    return Batch.from_data_list(batch)


def check_lp_feasibility(A_ub, b_ub):
    """
    Check the feasibility of a linear program using linprog from scipy.optimize.

    Args:
    c (numpy array): Coefficients for the objective function.
    A_ub (numpy array): Coefficients for the inequality constraints.
    b_ub (numpy array): Right-hand side vector for the inequality constraints.
    A_eq (numpy array): Coefficients for the equality constraints.
    b_eq (numpy array): Right-hand side vector for the equality constraints.

    Returns:
    bool: True if the LP is feasible, False otherwise.
    """
    res = linprog(
        np.zeros(A_ub.shape[1]),
        A_ub=A_ub,
        b_ub=b_ub,
        method="highs",
        bounds=(0, None),
    )

    return not res.status == 2


def check_sat_satisfiability(cnf_formula):
    """
    Check if the given CNF formula is satisfiable.
    Args:
    cnf_formula (list of list of int): CNF formula represented as a list of clauses,
                                       where each clause is a list of integers.
    Returns:
    bool: True if satisfiable, False otherwise.
    """
    solver = Solver(name="g4")
    solver.append_formula(cnf_formula)
    is_satisfiable = solver.solve()
    solver.delete()  # Clean up solver instance
    return is_satisfiable


def generate_random_cnf(num_clauses, num_variables, weight="const"):
    """
    Generates a random CNF formula.

    Args:
    num_clauses (int): Number of clauses in the CNF.
    max_literals (int): Maximum number of literals per clause.
    num_variables (int): Number of different variables used in the CNF.

    Returns:
    list of lists: A CNF formula represented as a list of clauses,
                   where each clause is a list of integers.
    """
    cnf = []

    for _ in range(num_clauses):
        num_literals_in_clause = random.randint(
            1, num_variables
        )  # At least 1 literal per clause
        clause = set()
        while len(clause) < num_literals_in_clause:
            variable = random.randint(1, num_variables)
            is_negated = random.choice([True, False])
            literal = -variable if is_negated else variable
            # Ensure no x and -x in the same clause
            if -literal not in clause:
                clause.add(literal)
        cnf.append(list(clause))
    if weight == "const":
        weight = np.ones(num_clauses)
    else:
        weight = np.random.uniform(0, 1, num_clauses)
    return cnf, weight


def cnf_to_matrix(cnf, num_variables):
    """
    Converts a CNF formula to a matrix representation.

    Args:
    cnf (list of lists): A CNF formula represented as a list of clauses,
                         where each clause is a list of integers.
    num_variables (int): Number of different variables used in the CNF.

    Returns:
    np.ndarray: A matrix of shape (num_clauses, num_variables) where
                each entry indicates the presence or absence of a literal.
    """
    num_clauses = len(cnf)
    # Initialize a matrix of zeros
    matrix = np.zeros((num_clauses, num_variables), dtype=int)

    # Fill the matrix according to the CNF formula
    for i, clause in enumerate(cnf):
        for literal in clause:
            variable_index = abs(literal) - 1  # Convert to zero-based index
            if literal > 0:
                matrix[i, variable_index] = 1  # Positive literal
            else:
                matrix[i, variable_index] = -1  # Negated literal
    return matrix


if __name__ == "__main__":
    # Parameters
    n = 2  # Number of variables
    m = 10  # Number of inequality constraints
    num_instances = 1000

    # Generate and check multiple instances
    feasible_count = 0
    infeasible_count = 0

    for _ in range(num_instances):
        (c, A_ub, b_ub, constraint_weight) = generate_random_lp(1, n, m)
        feasible = check_lp_feasibility(A_ub[0], b_ub[0])
        if feasible:
            feasible_count += 1
        else:
            infeasible_count += 1
    print(f"Feasible instances: {feasible_count}")
    print(f"Infeasible instances: {infeasible_count}")
    # batch_size, n, m = 1, 2, 4
    # c, A_ub, b_ub = generate_random_lp(batch_size, n, m)
    # bipartite_graphs = lp_to_graph(A_ub, b_ub, c)

    # # from bipartite import Encoder

    # print(bipartite_graphs.x_s.shape)
    # print(bipartite_graphs.x_t.shape)
    # print(bipartite_graphs.edge_attr.shape)
    # print(bipartite_graphs.edge_index)

    # # encoder = Encoder(1, 1, 2, 16)

    # # x_s, x_t = encoder(
    # #     bipartite_graphs.x_s,
    # #     bipartite_graphs.x_t,
    # #     bipartite_graphs.edge_index,
    # #     bipartite_graphs.edge_attr,
    # #     bipartite_graphs.batch,
    # # )
    # # print(x_s.shape, x_t.shape)
