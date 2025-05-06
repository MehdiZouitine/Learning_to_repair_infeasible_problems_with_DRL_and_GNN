import random
import numpy as np
from pysat.solvers import Solver
from tqdm import tqdm


def is_satisfiable(cnf):
    """Check if a given CNF formula is satisfiable using a SAT solver."""
    with Solver(name="g3") as solver:
        for clause in cnf:
            solver.add_clause(clause)
        return solver.solve()  # True if SAT, False if UNSAT


def generate_k_sat(N, L, k):
    """
    Generates a random k-SAT CNF formula.

    Args:
    N (int): Number of variables.
    L (int): Number of clauses.
    k (int): Number of literals per clause.

    Returns:
    list: CNF formula represented as a list of clauses.
    """
    cnf = []
    for _ in range(L):
        clause = set()
        while len(clause) < k:
            variable = random.randint(1, N)
            # variable = N
            is_negated = random.choice([True, False])
            literal = -variable if is_negated else variable
            if -literal not in clause:  # Avoid tautologies
                clause.add(literal)
        cnf.append(list(clause))
    return cnf


def generate_mixed_k_sat(N, L, k_max, seed=None):
    """
    Generates a random mixed k-SAT CNF formula, where clause lengths vary between 2 and N,
    ensuring no tautologies (no clause contains both x and ¬x).

    Args:
    N (int): Number of variables.
    L (int): Number of clauses.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    list: CNF formula represented as a list of clauses.
    """
    if seed is not None:
        random.seed(seed)

    cnf = []
    for _ in range(L):
        # k = random.randint(2, k_max)  # Random clause size between 2 and N
        k = k_max
        # Sample k distinct variables (without replacement)
        variables = random.sample(range(1, N + 1), k)

        # Randomly negate each variable
        clause = [var if random.choice([True, False]) else -var for var in variables]

        cnf.append(clause)

    return cnf


import matplotlib.pyplot as plt


def estimate_satisfiability_probability(N, L_ratios, M, k_max, mixed=False, seed=None):
    """
    Iterates over different L/N ratios and computes the probability of satisfiability.

    Args:
    N (int): Number of variables.
    L_ratios (list): List of L/N ratios to evaluate.
    M (int): Number of CNFs to generate per L/N ratio.
    k (int): Number of literals per clause (ignored if mixed=True).
    mixed (bool): If True, use mixed k-SAT instead of fixed k-SAT.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    None (Generates a plot of probability of SAT vs. L/N ratio)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    probabilities = []

    for ratio in tqdm(L_ratios):
        L = int(ratio * N)  # Compute L based on ratio
        sat_count = 0

        for _ in range(M):
            if mixed:
                cnf = generate_mixed_k_sat(N, L, k_max)

            # Check if CNF is satisfiable
            if is_satisfiable(cnf):
                sat_count += 1

        # Compute SAT probability
        prob_sat = sat_count / M
        probabilities.append(prob_sat)

    # Plot results
    plt.figure(figsize=(8, 5), dpi=600)
    plt.plot(
        L_ratios,
        probabilities,
        marker="o",
        linestyle="-",
        color="b",
        label="SAT Probability",
    )
    plt.axvline(
        x=4.26, color="r", linestyle="--", label="3-SAT Phase Transition (4.26)"
    )
    plt.xlabel("L/N Ratio")
    plt.ylabel("Probability of Satisfiability")
    plt.title(f"Satisfiability Probability vs. L/N Ratio (N={N}, M={M})")
    plt.legend()
    plt.grid(True)
    plt.savefig("satisfiability_probability.png")

if __name__ == "__main__":
    # Example usage
    N = 15  # Number of variables
    L_ratios = np.linspace(1, 10, 20)  # L/N values from 1 to 100
    M = 1000  # Number of CNFs per L/N ratio
    k_max = 3  # Fixed clause length
    estimate_satisfiability_probability(N, L_ratios, M, k_max, mixed=True, seed=None)
