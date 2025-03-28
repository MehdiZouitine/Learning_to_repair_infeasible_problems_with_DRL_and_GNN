import numpy as np
from scipy.optimize import linprog
from utils import (
    check_lp_feasibility,
    generate_random_lp,
    generate_random_infeasible_lp,
)
from scipy.optimize import milp, LinearConstraint, Bounds
from tqdm import tqdm
import time


def solve_elastic_lp(
    c,
    A_ub,
    b_ub,
    weights,
):
    """
    Solves the elastic LP problem by adjusting elastic variables according to inequality signs.

    Parameters:
    - c: Objective function coefficients (n,).
    - A_ub: Inequality constraints coefficients (m, n).
    - b_ub: Right-hand side of inequality constraints (m,).

    Returns:
    - result: The result of linprog solver, including solution status and other details.
    - Z: The optimal value of the objective function.
    """
    # Extend the LP to include elastic variables
    elastic_A_ub = np.concatenate((A_ub.copy(), -np.eye(len(b_ub.copy()))), axis=1)
    elastic_c = np.concatenate((c * 0, weights))

    result = linprog(
        elastic_c,
        A_ub=elastic_A_ub,
        b_ub=b_ub,
        method="highs",
        bounds=(0, None),
    )

    Z = result.fun
    return result, Z


def elasticfilter(A_ub, b_ub, weights):
    """
    Placeholder for the elasticfilter function.
    This should be replaced with the actual implementation.
    """
    result, fval = solve_elastic_lp(np.zeros(A_ub.shape[1]), A_ub, b_ub, weights)
    return result, fval


def create_candidate_set(result, activeA, idx2, k=None, weights=None):
    """
    Creates a candidate set of inequality indices based on the elastic variables.

    Parameters:
    - result: The result of the elastic LP solver, containing solution values.
    - activeA: Boolean array indicating active inequalities.
    - idx2: Indices of currently active inequalities.
    - weights: Weights of the constraints.

    Returns:
    - holdset: List of indices for the next candidate set.
    """
    elastic_variables = result.x[-activeA.sum() :]
    ineq_iis = idx2[elastic_variables > 0]
    holdset = ineq_iis.tolist()
    return holdset


def create_candidate_set_1(result, activeA, idx2, k=5, weights=None):
    """
    Creates a candidate set of inequality indices based on the violated constraints approach.

    Parameters:
    - result: The result of the elastic LP solver, containing solution values.
    - activeA: Boolean array indicating active inequalities.
    - idx2: Indices of currently active inequalities.
    - k: Limit for the number of candidate constraints to be considered.
    - weights: Weights of the constraints.

    Returns:
    - holdset: List of indices for the next candidate set.
    """
    elastic_variables = result.x[-activeA.sum() :]
    dual_prices = result.ineqlin["marginals"][-activeA.sum() :]

    # Compute the product for violated constraints
    violated_constraints = [
        (idx2[i], weights[idx2[i]] * elastic_variables[i] * abs(dual_prices[i]))
        for i in range(len(elastic_variables))
        if elastic_variables[i] > 0
    ]

    # Sort by product in descending order and take the top k
    violated_constraints = sorted(violated_constraints, key=lambda x: -x[1])[:k]
    holdset = [i[0] for i in violated_constraints]

    return holdset


def create_candidate_set_2(result, activeA, idx2, k=5, weights=None):
    """
    Creates a candidate set of inequality indices based on the satisfied constraints approach.

    Parameters:
    - result: The result of the elastic LP solver, containing solution values.
    - activeA: Boolean array indicating active inequalities.
    - idx2: Indices of currently active inequalities.
    - k: Limit for the number of candidate constraints to be considered.
    - weights: Weights of the constraints.

    Returns:
    - holdset: List of indices for the next candidate set.
    """
    elastic_variables = result.x[-activeA.sum() :]
    dual_prices = result.ineqlin["marginals"][-activeA.sum() :]

    # Compute the absolute dual prices for satisfied constraints
    satisfied_constraints = [
        (idx2[i], weights[idx2[i]] * abs(dual_prices[i]))
        for i in range(len(elastic_variables))
        if elastic_variables[i] == 0 and abs(dual_prices[i]) > 0
    ]

    # Sort by absolute dual price in descending order and take the top k
    satisfied_constraints = sorted(satisfied_constraints, key=lambda x: -x[1])[:k]
    holdset = [i[0] for i in satisfied_constraints]

    return holdset


def generate_cover(A_ub, b_ub, create_candidate_set_func, k=5, weights=None):
    """
    Returns the cover set of linear inequalities and the total number of calls to linprog.

    Parameters:
    - A_ub: Inequality constraints coefficients (m, n).
    - b_ub: Right-hand side of inequality constraints (m,).
    - create_candidate_set_func: Callable function to create the candidate set.
    - k: Limit for the number of candidate constraints to be considered.
    - weights: Weights of the constraints.

    Returns:
    - coverset: Indices of the cover set inequalities.
    - nlp: Total number of calls to linprog.
    """

    coverset = []
    activeA = np.ones(len(b_ub), dtype=bool)
    result, fval = elasticfilter(A_ub, b_ub, weights)
    nlp = 1

    # Use the provided candidate set creation function
    holdset = create_candidate_set_func(
        result, activeA, np.arange(len(b_ub)), k, weights
    )

    if len(holdset) == 1:
        return holdset, nlp

    candidate = holdset.copy()

    # Step 2 of Algorithm 7.3
    while len(candidate) > 0:
        minsinf = float("inf")
        winner = None
        nextwinner = []

        for i in range(len(candidate)):
            activeA[candidate[i]] = False
            idx2 = np.where(activeA)[0]

            result, fval = elasticfilter(
                A_ub[activeA, :], b_ub[activeA], weights[activeA]
            )
            nlp += 1

            if fval == 0:
                coverset.append(candidate[i])
                return coverset, nlp

            if fval < minsinf:
                winner = candidate[i]
                minsinf = fval
                holdset = create_candidate_set_func(
                    result, activeA, idx2, k, weights
                )  # Call the provided function

                if len(holdset) == 1:
                    nextwinner = holdset

            activeA[candidate[i]] = True

        if winner is not None:
            coverset.append(winner)
            activeA[winner] = False
            if nextwinner:
                coverset.extend(nextwinner)
                return coverset, nlp

        candidate = holdset.copy()

    return coverset, nlp


def big_m_coverset(A, b, weights):
    """
    Implement the big M method to find the MIN IIS COVER (coverset).

    Parameters:
    - A: Coefficient matrix for inequality constraints (m x n)
    - b: Right-hand side vector for inequality constraints (m)
    - weights: Weights of the constraints

    Returns:
    - coverset: Indices of constraints in the MIN IIS COVER
    """
    m, n = A.shape
    M = 1e6  # Big M value, adjust as needed

    # Objective: Minimize sum of y variables, considering weights
    c = np.zeros(n + m)
    c[n:] = weights

    # Constraints matrix
    A_constraint = np.hstack([A, -M * np.eye(m)])
    b_constraint = b

    # Variable bounds
    lb = np.zeros(n + m)
    ub = np.concatenate([np.full(n, np.inf), np.ones(m)])
    constraints = LinearConstraint(A_constraint, ub=b_constraint)
    # the lower bound is zero for all variables and the upper bound is infinity for the first n variables and 1 for the rest
    bounds = Bounds(lb, ub)
    # Integrality constraints
    integrality = np.concatenate([np.zeros(n), np.ones(m)])

    # Solve the MILP
    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )

    # Extract the coverset
    y_values = res.x[n:]
    coverset = np.where(y_values > 0.5)[0]  # Use 0.5 as threshold for binary variables

    return coverset, None


def grow(A, b, weights):
    """Start with a single constraint, and add constraints one by one.
    When a newly added constraint triggers infeasibility, discard it.
    This is the grow method used by Bailey and Stuckey (2005) to find maximal feasible subsystems.
    """
    # use check_feasibility to check feasibility
    m, n = A.shape
    constraint_set = set()
    coverset = []
    for i in range(m):
        constraint_set.add(i)
        current_A = A[list(constraint_set)]
        current_b = b[list(constraint_set)]
        if not check_lp_feasibility(current_A, current_b):
            constraint_set.remove(i)
            coverset.append(i)
    return coverset, m


def compute_baseline(batch_size, c, v, method_name, weights="const"):
    """
    Computes the performance and time taken for a specific method over a batch of instances.

    Parameters:
    - batch_size: The number of LP instances to test.
    - method_name: The name of the method to evaluate
                   ("grow", "big_m", "generate_cover_set_1", or "generate_cover_set_2").

    Returns:
    - avg_size: Average size of the cover set.
    - std_size: Standard deviation of the cover set size.
    - avg_time: Average time taken per instance.
    """
    size = []
    times = []
    nlp = []

    # Define the method based on the method_name
    if method_name == "Deletion":
        method = grow
    elif method_name == "Opt":
        method = big_m_coverset
    elif method_name == "Chinneck":

        def method(A_ub, b_ub, weights):
            return generate_cover(
                A_ub, b_ub, create_candidate_set, k=2, weights=weights
            )
    elif method_name == "Chinneck-Fast":

        def method(A_ub, b_ub, weights):
            return generate_cover(
                A_ub, b_ub, create_candidate_set_1, k=2, weights=weights
            )
    else:
        raise ValueError("Invalid method name provided.")

    for j in tqdm(range(batch_size)):
        # Generate random infeasible LP
        c_batch, A_ub_batch, b_ub_batch, constraint_weight_batch = (
            generate_random_infeasible_lp(1, v, c, weight=weights, seed=j)
        )
        _, A_ub, b_ub, constraint_weight = (
            c_batch[0],
            A_ub_batch[0],
            b_ub_batch[0],
            constraint_weight_batch[0],
        )

        # Measure time taken for the method
        start = time.time()
        coverset, current_nlp = method(A_ub, b_ub, constraint_weight)
        end = time.time()

        size.append(constraint_weight[coverset].sum())
        times.append(end - start)
        nlp.append(current_nlp)

    avg_size = np.mean(size)
    std_size = np.std(size)
    avg_time = np.mean(times)
    if method_name == "big_m":
        avg_nlp = None
    else:
        avg_nlp = np.mean(nlp)

    return avg_size, std_size, avg_time, avg_nlp


if __name__ == "__main__":
    from tqdm import tqdm
    import time
    import pandas as pd

    instance_sizes = [
        (10, 2),
        (20, 5),
        (50, 10),
        (100, 20),
        (150, 30),
        (200, 40),
        (300, 60),
    ]
    instance_sizes_name = [
        "c10v2",
        "c20v5",
        "c50v10",
        "c100v20",
        "c150v30",
        "c200v40",
        "c300v60",
    ]
    methods = ["Chinneck", "Chinneck-Fast"]
    batch_size = 10
    # create a pandas dataset where each row corresponds to an instances size and each column corresponds to a method
    data = pd.DataFrame(columns=methods, index=instance_sizes_name)
    for m in methods:
        for c, v in instance_sizes:
            avg_size, std_size, avg_time, avg_nlp = compute_baseline(
                batch_size, c, v, m, weights="uniform"
            )
            data.loc[f"c{c}v{v}", m] = avg_nlp
            print(f"Method: {m}, Instance: c{c}v{v} : Avg nlp: {avg_nlp}")
    data.to_csv("avg_nlp.csv")
