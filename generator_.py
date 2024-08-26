import numpy as np
from scipy.optimize import linprog
import torch

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected


class BipartiteData(Data):
    def __init__(
        self, edge_index=None, x_s=None, x_t=None, edge_attr=None, num_nodes=None
    ):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    @property
    def num_nodes(self):
        return self.x_s.size(0) + self.x_t.size(0)


def generate_random_lp(batch_size, n, m, seed=None):
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
    integrality = np.zeros((batch_size, n))

    return c, A_ub, b_ub, integrality


def generate_random_ilp(batch_size, n, m, seed=None):
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
    integrality = np.ones((batch_size, n))
    return c, A_ub, b_ub, integrality


def generate_random_milp(batch_size, n, m, seed=None):
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
    integrality = np.random.randint(0, 2, (batch_size, n))
    return c, A_ub, b_ub, integrality


def lp_to_matrix(A_ub, b_ub, c, integrality):
    integrality = integrality[:, np.newaxis, :]
    integrality = np.repeat(integrality, A_ub.shape[1], axis=1)
    integrality = integrality[:, :, :, np.newaxis]
    A_ub = A_ub[:, :, :, np.newaxis]
    A_ub = np.concatenate(
        (A_ub, np.zeros((A_ub.shape[0], A_ub.shape[1], A_ub.shape[2], 1))), axis=-1
    )

    b_ub = b_ub[:, :, np.newaxis, np.newaxis]
    b_ub = np.repeat(b_ub, A_ub.shape[2], axis=2)

    c = c[:, np.newaxis, :, np.newaxis]
    c = np.repeat(c, A_ub.shape[1], axis=1)

    matrix = np.concatenate((A_ub, b_ub, c, integrality), axis=-1)
    return matrix / 100


def lp_to_graph(A_ub, b_ub, c):
    batch_size, num_constraints, num_variables = A_ub.shape

    bipartite_graphs = []

    # Generate edge indices for a fully connected bipartite graph
    variable_indices = torch.arange(num_variables)
    constraint_indices = torch.arange(num_constraints)
    edge_index = (
        torch.cartesian_prod(variable_indices, constraint_indices).t().contiguous()
    )

    for batch in range(batch_size):
        A = A_ub[batch]
        b = b_ub[batch]
        c_vector = c[batch]

        # Node features
        x_s = torch.tensor(c_vector[:, np.newaxis], dtype=torch.float)  # Variable nodes
        x_t = torch.tensor(b[:, np.newaxis], dtype=torch.float)  # Constraint nodes
        # Edge attributes (flatten the A matrix)
        edge_attr = torch.tensor(A.flatten(), dtype=torch.float).unsqueeze(1)
        # Create a bipartite graph
        bipartite_graph = BipartiteData(
            edge_index=edge_index,
            x_s=x_s / 100,
            x_t=x_t / 100,
            edge_attr=edge_attr / 100,
            num_nodes=num_variables + num_constraints,
        )
        bipartite_graphs.append(bipartite_graph)

    return Batch.from_data_list(bipartite_graphs)


def check_feasibility(c, A_ub, b_ub, integrality):
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
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        method="highs",
        bounds=(0, None),
        integrality=integrality,
    )

    return not res.status == 2


# from ortools.linear_solver import pywraplp


# def check_feasibility(c, A_ub, b_ub):
#     """
#     Check the feasibility of a linear program using OR-Tools.

#     Args:
#     c (numpy array): Coefficients for the objective function.
#     A_ub (numpy array): Coefficients for the inequality constraints.
#     b_ub (numpy array): Right-hand side vector for the inequality constraints.

#     Returns:
#     bool: True if the LP is feasible, False otherwise.
#     """
#     # Create the linear solver with GLOP backend
#     solver = pywraplp.Solver.CreateSolver("GLOP")

#     if not solver:
#         raise Exception("Solver not created.")

#     # Number of variables
#     num_vars = len(c)

#     # Add variables with bounds (-100, 100)
#     x = [solver.NumVar(-100, 100, f"x_{i}") for i in range(num_vars)]

#     # Add inequality constraints A_ub * x <= b_ub
#     for i in range(A_ub.shape[0]):
#         constraint = solver.RowConstraint(
#             -solver.infinity(), b_ub[i], f"constraint_{i}"
#         )
#         for j in range(num_vars):
#             constraint.SetCoefficient(x[j], A_ub[i, j])

#     # Define the objective function (can be anything for feasibility check)
#     objective = solver.Objective()
#     for j in range(num_vars):
#         objective.SetCoefficient(x[j], c[j])
#     objective.SetMinimization()

#     # Solve the linear program
#     status = solver.Solve()

#     # Check if the solution is feasible
#     return status == pywraplp.Solver.FEASIBLE or status == pywraplp.Solver.OPTIMAL


def compute_cost(action, A_ub, b_ub, c):
    A_ub_action = A_ub[action.astype(bool), :]
    b_ub_action = b_ub[action.astype(bool)]
    if check_feasibility(c, A_ub_action, b_ub_action):
        return 1
    return -1


if __name__ == "__main__":
    # Parameters
    n = 2  # Number of variables
    m = 20  # Number of inequality constraints
    num_instances = 1000

    # Generate and check multiple instances
    feasible_count = 0
    infeasible_count = 0

    for _ in range(num_instances):
        (c, A_ub, b_ub, integrality) = generate_random_milp(1, n, m)
        if check_feasibility(c[0], A_ub[0], b_ub[0], integrality[0]):
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
