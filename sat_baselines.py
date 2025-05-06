from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
import random
import numpy as np
from utils import check_sat_satisfiability
from typing import List, Tuple


def maxsat(cnf_formula: List[List[int]], weights: np.ndarray) -> float:
    """
    Solves the weighted MaxSAT problem using RC2.

    Args:
        cnf_formula: CNF formula as a list of clauses (each clause is a list of integers).
        weights: Clause weights.

    Returns:
        Cost of the optimal MaxSAT solution (minimal total weight of removed clauses).
    """
    wcnf = WCNF()
    for idx, clause in enumerate(cnf_formula):
        wcnf.append(clause, weight=weights[idx])
    with RC2(wcnf) as rc2:
        rc2.compute()
    return rc2.cost


def maxsat_random(cnf_formula: List[List[int]], weights: np.ndarray) -> float:
    """
    Random MaxSAT baseline:
    Removes random clauses until the formula becomes satisfiable.

    Args:
        cnf_formula: CNF formula.
        weights: Clause weights.

    Returns:
        Total weight of removed clauses.
    """
    weights = weights.tolist()
    removed_clauses = []
    cnf_formula = cnf_formula.copy()

    while not check_sat_satisfiability(cnf_formula):
        idx = random.randint(0, len(cnf_formula) - 1)
        removed_clauses.append((cnf_formula.pop(idx), weights.pop(idx)))

    return np.sum([r[1] for r in removed_clauses])


if __name__ == "__main__":
    from utils import generate_random_infeasible_cnf
    from tqdm import tqdm

    batch_size = 300
    k_max = 3
    num_clauses = 91
    num_variables = 20
    all_cost = []
    all_cost_random = []

    k = 0
    while k < batch_size:
        cnf, weight = generate_random_infeasible_cnf(
            num_clauses, num_variables, weight="const", seed=k, k_max=k_max
        )
        if check_sat_satisfiability(cnf):
            continue
        k += 1
        cost = maxsat(cnf, weight)
        cost_random = maxsat_random(cnf, weight)
        all_cost.append(cost)
        all_cost_random.append(cost_random)

    print(f"Average cost (RC2): {np.mean(all_cost):.2f} ± {np.std(all_cost):.2f}")
    print(f"Average cost random: {np.mean(all_cost_random):.2f} ± {np.std(all_cost_random):.2f}")
