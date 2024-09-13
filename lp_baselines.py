import numpy as np
from scipy.optimize import linprog
from utils import (
    check_lp_feasibility,
    generate_random_lp,
    generate_random_infeasible_lp,
)
from scipy.optimize import milp, LinearConstraint, Bounds


def create_candidate_set(result, activeA, idx2, k=None):
    """
    Creates a candidate set of inequality indices based on the elastic variables.

    Parameters:
    - result: The result of the elastic LP solver, containing solution values.
    - activeA: Boolean array indicating active inequalities.
    - idx2: Indices of currently active inequalities.

    Returns:
    - holdset: List of indices for the next candidate set.
    """
    elastic_variables = result.x[-activeA.sum() :]
    ineq_iis = idx2[elastic_variables > 0]
    holdset = ineq_iis.tolist()
    return holdset


def create_candidate_set_1(result, activeA, idx2, k=5):
    """
    Creates a candidate set of inequality indices based on the violated constraints approach.

    Parameters:
    - result: The result of the elastic LP solver, containing solution values.
    - activeA: Boolean array indicating active inequalities.
    - idx2: Indices of currently active inequalities.
    - k: Limit for the number of candidate constraints to be considered.

    Returns:
    - holdset: List of indices for the next candidate set.
    """
    elastic_variables = result.x[-activeA.sum() :]
    dual_prices = result.ineqlin["marginals"][-activeA.sum() :]

    # Compute the product for violated constraints
    violated_constraints = [
        (idx2[i], elastic_variables[i] * abs(dual_prices[i]))
        for i in range(len(elastic_variables))
        if elastic_variables[i] > 0
    ]

    # Sort by product in descending order and take the top k
    violated_constraints = sorted(violated_constraints, key=lambda x: -x[1])[:k]
    holdset = [i[0] for i in violated_constraints]

    return holdset


def create_candidate_set_2(result, activeA, idx2, k=5):
    """
    Creates a candidate set of inequality indices based on the satisfied constraints approach.

    Parameters:
    - result: The result of the elastic LP solver, containing solution values.
    - activeA: Boolean array indicating active inequalities.
    - idx2: Indices of currently active inequalities.
    - k: Limit for the number of candidate constraints to be considered.

    Returns:
    - holdset: List of indices for the next candidate set.
    """
    elastic_variables = result.x[-activeA.sum() :]
    dual_prices = result.ineqlin["marginals"][-activeA.sum() :]

    # Compute the absolute dual prices for satisfied constraints
    satisfied_constraints = [
        (idx2[i], abs(dual_prices[i]))
        for i in range(len(elastic_variables))
        if elastic_variables[i] == 0 and abs(dual_prices[i]) > 0
    ]

    # Sort by absolute dual price in descending order and take the top k
    satisfied_constraints = sorted(satisfied_constraints, key=lambda x: -x[1])[:k]
    holdset = [i[0] for i in satisfied_constraints]

    return holdset


def generate_cover(A_ub, b_ub, create_candidate_set_func, k=5):
    """
    Returns the cover set of linear inequalities and the total number of calls to linprog.

    Parameters:
    - A_ub: Inequality constraints coefficients (m, n).
    - b_ub: Right-hand side of inequality constraints (m,).
    - create_candidate_set_func: Callable function to create the candidate set.
    - k: Limit for the number of candidate constraints to be considered.

    Returns:
    - coverset: Indices of the cover set inequalities.
    - nlp: Total number of calls to linprog.
    """

    coverset = []
    activeA = np.ones(len(b_ub), dtype=bool)
    result, fval = elasticfilter(A_ub, b_ub)
    nlp = 1

    # Use the provided candidate set creation function
    holdset = create_candidate_set_func(result, activeA, np.arange(len(b_ub)), k)

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

            result, fval = elasticfilter(A_ub[activeA, :], b_ub[activeA])
            nlp += 1

            if fval == 0:
                coverset.append(candidate[i])
                return coverset, nlp

            if fval < minsinf:
                winner = candidate[i]
                minsinf = fval
                holdset = create_candidate_set_func(
                    result, activeA, idx2, k
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


def solve_elastic_lp(c, A_ub, b_ub):
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
    elastic_c = np.concatenate((c * 0, np.ones(len(b_ub))))

    result = linprog(
        elastic_c,
        A_ub=elastic_A_ub,
        b_ub=b_ub,
        method="highs",
        bounds=(0, None),
    )

    Z = result.fun
    return result, Z


def elasticfilter(A_ub, b_ub):
    """
    Placeholder for the elasticfilter function.
    This should be replaced with the actual implementation.
    """
    result, fval = solve_elastic_lp(np.zeros(A_ub.shape[1]), A_ub, b_ub)
    return result, fval


def big_m_coverset(A, b):
    """
    Implement the big M method to find the MIN IIS COVER (coverset).

    Parameters:
    - A: Coefficient matrix for inequality constraints (m x n)
    - b: Right-hand side vector for inequality constraints (m)

    Returns:
    - coverset: Indices of constraints in the MIN IIS COVER
    """
    m, n = A.shape
    M = 1e6  # Big M value, adjust as needed

    # Objective: Minimize sum of y variables
    c = np.zeros(n + m)
    c[n:] = 1

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

    return coverset


def grow(A, b):
    """Start with a single constraint, and add constraints one by one.
    When a newly added constraint triggers infeasibility, discard it.
    This is the grow method used by Bailey and Stuckey (2005) to find maximal fea- sible subsystems.
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


if __name__ == "__main__":
    from tqdm import tqdm
    import time

    # Example usage:
    batch_size = 300
    n = 10  # number of variables
    m = 50  # number of constraints
    exact_size = []
    size = []
    n_calls = []
    start = time.time()
    for j in tqdm(range(batch_size)):
        c_batch, A_ub_batch, b_ub_batch, integrality_batch = (
            generate_random_infeasible_lp(1, n, m, seed=j)
        )
        c, A_ub, b_ub, _ = (
            c_batch[0],
            A_ub_batch[0],
            b_ub_batch[0],
            integrality_batch[0],
        )
        # print("Initial Feasibility:", check_feasibility(c, A_ub, b_ub, integrality))
        coverset, n_call = generate_cover(A_ub, b_ub, create_candidate_set, k=2)
        # coverset, n_call = grow(A_ub, b_ub, integrality)
        size.append(len(coverset))

        # n_calls.append(n_call)

        # # Remove constraints in the cover set
        # A_ub_new = np.delete(A_ub, coverset, axis=0)
        # b_ub_new = np.delete(b_ub, coverset)
        # # print("heuristic coverset length:", len(coverset))
        # print(
        #     "Reduced Feasibility:",
        #     check_feasibility(c, A_ub_new, b_ub_new, integrality),
        # )

        # exact_coverset = big_m_coverset(A_ub, b_ub)
        # exact_size.append(len(exact_coverset))
        # # print("exact coverset length:", len(exact_coverset))
        # A_ub_new_exact = np.delete(A_ub, exact_coverset, axis=0)
        # b_ub_new_exact = np.delete(b_ub, exact_coverset)

        # print("Number of calls:", n_call)
        # print("Length of max_fs:", len(coverset))

        # print(
        #     "Reduced Feasibility exact:",
        #     check_feasibility(c, A_ub_new_exact, b_ub_new_exact, integrality),
        # )
        # print("#############################")
        # input()  # Pause to inspect each iteration if necessary

    print(np.mean(size), np.std(size))
    # print(np.mean(n_calls), np.std(n_calls))
    # print("Time taken:", (time.time() - start) / batch_size)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(np.mean(exact_size), np.std(exact_size))
