from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
import random
from utils import check_sat_satisfiability


def maxsat(cnf_formula, weights):
    wcnf = WCNF()
    for idx, clause in enumerate(cnf_formula):
        wcnf.append(clause, weight=weights[idx])
    with RC2(wcnf) as rc2:
        rc2.compute()
    return rc2.cost


def maxsat_random(cnf_formula, weights):
    # remove one clause at random until the formula is satisfiable
    weights = weights.tolist()
    removed_clauses = []
    cnf_formula = cnf_formula.copy()
    while not check_sat_satisfiability(cnf_formula):
        idx = random.randint(0, len(cnf_formula) - 1)
        removed_clauses.append((cnf_formula.pop(idx), weights.pop(idx)))
    return np.sum([r[1] for r in removed_clauses])


if __name__ == "__main__":
    from utils import check_sat_satisfiability, generate_random_infeasible_cnf
    import numpy as np
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
    print(f"Average cost: {np.mean(all_cost)} std: {np.std(all_cost)}")
    print(
        f"Average cost random: {np.mean(all_cost_random)} std: {np.std(all_cost_random)}"
    )
