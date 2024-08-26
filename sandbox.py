import numpy as np
import numpy as np
from scipy.optimize import linprog
from generator_ import (
    check_feasibility,
    generate_random_lp,
    generate_random_ilp,
    generate_random_milp,
)

import numpy as np
from scipy.optimize import linprog


def generate_cover(A_ub, b_ub, integrality):
    """
    Returns the cover set of linear inequalities and the total number of calls to linprog.

    Parameters:
    - A_ub: Inequality constraints coefficients (m, n).
    - b_ub: Right-hand side of inequality constraints (m,).
    - lb: Lower bounds for variables (optional).
    - ub: Upper bounds for variables (optional).

    Returns:
    - coverset: Indices of the cover set inequalities.
    - nlp: Total number of calls to linprog.
    """

    coverset = []
    activeA = np.ones(len(b_ub), dtype=bool)

    # Step 1 of Algorithm 7.3
    result, fval = elasticfilter(A_ub, b_ub, integrality)
    nlp = 1
    elastic_variables = result.x[-len(b_ub) :]
    ninf = len([i for i in elastic_variables if i > 0])
    holdset = [i for i in range(len(b_ub)) if elastic_variables[i] > 0]
    if ninf == 1:
        return holdset, nlp

    candidate = holdset.copy()

    # Step 2 of Algorithm 7.3
    while np.sum(candidate) > 0:
        minsinf = float("inf")
        winner = None
        nextwinner = []

        for i in range(len(candidate)):
            activeA[candidate[i]] = False
            idx2 = np.where(activeA)[0]

            result, fval = elasticfilter(A_ub[activeA, :], b_ub[activeA], integrality)
            elastic_variables = result.x[-activeA.sum() :]
            nlp += 1
            ineq_iis = idx2[elastic_variables > 0]

            if fval == 0:
                coverset.append(candidate[i])
                return coverset, nlp

            if fval < minsinf:
                winner = candidate[i]
                minsinf = fval
                holdset = ineq_iis.tolist()

                if len(ineq_iis) == 1:
                    nextwinner = ineq_iis.tolist()

            activeA[candidate[i]] = True

        # Step 3 of Algorithm 7.3
        if winner is not None:
            coverset.append(winner)
            activeA[winner] = False
            if nextwinner:
                coverset.extend(nextwinner)
                return coverset, nlp

        candidate = holdset.copy()

    return coverset, nlp


def solve_elastic_lp(c, A_ub, b_ub, integrality):
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
        integrality=np.concatenate((integrality, np.zeros(len(b_ub)))),
    )

    Z = result.fun
    return result, Z


def elasticfilter(A_ub, b_ub, integrality):
    """
    Placeholder for the elasticfilter function.
    This should be replaced with the actual implementation.
    """
    result, fval = solve_elastic_lp(
        np.zeros(A_ub.shape[1]), A_ub, b_ub, integrality=integrality
    )
    return result, fval


if __name__ == "__main__":
    from tqdm import tqdm
    from generator_ import check_feasibility, generate_random_lp
    import time

    # Example usage:
    batch_size = 300
    n = 5  # number of variables
    m = 30  # number of constraints

    size = []
    n_calls = []
    start = time.time()
    for j in tqdm(range(batch_size)):
        c_batch, A_ub_batch, b_ub_batch, integrality_batch = generate_random_milp(
            1, n, m, seed=j
        )
        c, A_ub, b_ub, integrality = (
            c_batch[0],
            A_ub_batch[0],
            b_ub_batch[0],
            integrality_batch[0],
        )
        # print("Initial Feasibility:", check_feasibility(c, A_ub, b_ub, integrality))
        coverset, n_call = generate_cover(A_ub, b_ub, integrality)
        n_calls.append(n_call)

        # Remove constraints in the cover set
        A_ub_new = np.delete(A_ub, coverset, axis=0)
        b_ub_new = np.delete(b_ub, coverset)
        # print(len(coverset))
        size.append(len(coverset))

        # print("Number of calls:", n_call)
        # print("Length of max_fs:", len(coverset))

        # print(
        #     "Reduced Feasibility:",
        #     check_feasibility(c, A_ub_new, b_ub_new, integrality),
        # )
        # input()  # Pause to inspect each iteration if necessary

    print(np.mean(size), np.std(size))
    print(np.mean(n_calls), np.std(n_calls))
    print("Time taken:", (time.time() - start) / batch_size)
