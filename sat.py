from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def maxsat(cnf_formula, weights):
    wcnf = WCNF()
    for idx, clause in enumerate(cnf_formula):
        wcnf.append(clause, weight=weights[idx])
    with RC2(wcnf) as rc2:
        rc2.compute()
    return rc2.cost


if __name__ == "__main__":
    from utils import check_sat_satisfiability, generate_random_cnf
    import numpy as np
    from tqdm import tqdm

    batch_size = 300
    num_clauses = 150
    num_variables = 10
    all_cost = []
    k = 0
    while k < batch_size:
        cnf, weight = generate_random_cnf(num_clauses, num_variables)
        if check_sat_satisfiability(cnf):
            continue
        k += 1
        cost = maxsat(cnf, weight)
        all_cost.append(cost)
    print(np.mean(all_cost))
    print(np.std(all_cost))
