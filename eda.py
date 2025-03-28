from pysat.formula import CNF
from sat import maxsat, check_sat_satisfiability
import glob
import numpy as np
from tqdm import tqdm


def read_cnf_pysat(filename):
    """
    Reads a .cnf file using PySAT and returns the CNF formula.

    Args:
    filename (str): Path to the .cnf file.

    Returns:
    CNF object: PySAT CNF object containing clauses.
    """
    cnf = CNF(from_file=filename)
    return cnf


# Example usage
all_cost = []
for filename in tqdm(glob.glob("data/UUF250.1065.100/*.cnf")):
    cnf = read_cnf_pysat(filename)

    # Print extracted information
    # print(f"Number of variables: {cnf.nv}")
    # print(f"Number of clauses: {len(cnf.clauses)}")
    # print("CNF Clauses:", cnf.clauses[:5])  # Show first 5 clauses

    print(check_sat_satisfiability(cnf.clauses))
#     clauses = [c for c in cnf.clauses if c]
#     cost = maxsat(clauses, [1] * len(clauses))
#     all_cost.append(cost)
# print(f"Average cost: {np.mean(all_cost)} std: {np.std(all_cost)}")
