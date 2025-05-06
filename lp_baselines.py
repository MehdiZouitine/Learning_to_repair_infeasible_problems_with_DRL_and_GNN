import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from utils import (
    check_lp_feasibility,
    generate_random_lp,
    generate_random_infeasible_lp,
)
from tqdm import tqdm
import time
from typing import Callable, List, Tuple, Optional


def solve_elastic_lp(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    weights: np.ndarray
) -> Tuple[linprog, float]:
    """
    Solve an elastic linear program by adding slack variables.

    Args:
        c: Objective function coefficients (n,).
        A_ub: Inequality constraint matrix (m, n).
        b_ub: Constraint right-hand side (m,).
        weights: Slack variable penalties (m,).

    Returns:
        result: Optimization result from scipy linprog.
        Z: Objective value at the solution.
    """
    elastic_A_ub = np.concatenate((A_ub, -np.eye(len(b_ub))), axis=1)
    elastic_c = np.concatenate((c * 0, weights))

    result = linprog(
        elastic_c,
        A_ub=elastic_A_ub,
        b_ub=b_ub,
        method="highs",
        bounds=(0, None),
    )
    return result, result.fun


def elasticfilter(
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    weights: np.ndarray
) -> Tuple[linprog, float]:
    """
    Run an elastic LP feasibility filter.

    Args:
        A_ub: Inequality matrix.
        b_ub: RHS vector.
        weights: Slack penalties.

    Returns:
        result: linprog output.
        fval: Objective value.
    """
    return solve_elastic_lp(
        c=np.zeros(A_ub.shape[1]),
        A_ub=A_ub,
        b_ub=b_ub,
        weights=weights
    )


def create_candidate_set(
    result: linprog,
    activeA: np.ndarray,
    idx2: np.ndarray,
    k: Optional[int] = None,
    weights: Optional[np.ndarray] = None
) -> List[int]:
    """
    Find constraints with positive slack variables.

    Args:
        result: linprog output.
        activeA: Active constraints mask.
        idx2: Active indices.
        k: Not used.
        weights: Not used.

    Returns:
        List of violated constraint indices.
    """
    elastic_vars = result.x[-activeA.sum():]
    return idx2[elastic_vars > 0].tolist()


def create_candidate_set_1(
    result: linprog,
    activeA: np.ndarray,
    idx2: np.ndarray,
    k: int = 5,
    weights: Optional[np.ndarray] = None
) -> List[int]:
    """
    Select top-k most violated constraints.

    Returns:
        Indices of k most violated constraints.
    """
    elastic_vars = result.x[-activeA.sum():]
    dual_prices = result.ineqlin["marginals"][-activeA.sum():]

    violated = [
        (idx2[i], weights[idx2[i]] * elastic_vars[i] * abs(dual_prices[i]))
        for i in range(len(elastic_vars))
        if elastic_vars[i] > 0
    ]

    violated = sorted(violated, key=lambda x: -x[1])[:k]
    return [i[0] for i in violated]


def create_candidate_set_2(
    result: linprog,
    activeA: np.ndarray,
    idx2: np.ndarray,
    k: int = 5,
    weights: Optional[np.ndarray] = None
) -> List[int]:
    """
    Select top-k satisfied constraints with largest dual prices.

    Returns:
        Indices of k satisfied constraints.
    """
    elastic_vars = result.x[-activeA.sum():]
    dual_prices = result.ineqlin["marginals"][-activeA.sum():]

    satisfied = [
        (idx2[i], weights[idx2[i]] * abs(dual_prices[i]))
        for i in range(len(elastic_vars))
        if elastic_vars[i] == 0 and abs(dual_prices[i]) > 0
    ]

    satisfied = sorted(satisfied, key=lambda x: -x[1])[:k]
    return [i[0] for i in satisfied]


def generate_cover(
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    create_candidate_set_func: Callable,
    k: int = 5,
    weights: Optional[np.ndarray] = None
) -> Tuple[List[int], int]:
    """
    Compute a cover set using a deletion-based iterative algorithm.

    Args:
        A_ub: Inequality matrix.
        b_ub: RHS vector.
        create_candidate_set_func: Function to select candidate constraints.
        k: Number of candidates per iteration.
        weights: Constraint weights.

    Returns:
        coverset: Selected cover set indices.
        nlp: Number of LP solves performed.
    """
    coverset = []
    activeA = np.ones(len(b_ub), dtype=bool)
    result, _ = elasticfilter(A_ub, b_ub, weights)
    nlp = 1

    holdset = create_candidate_set_func(result, activeA, np.arange(len(b_ub)), k, weights)

    if len(holdset) == 1:
        return holdset, nlp

    candidate = holdset.copy()

    while candidate:
        minsinf = float("inf")
        winner = None
        nextwinner = []

        for i in candidate:
            activeA[i] = False
            idx2 = np.where(activeA)[0]

            result, fval = elasticfilter(
                A_ub[activeA, :], b_ub[activeA], weights[activeA]
            )
            nlp += 1

            if fval == 0:
                coverset.append(i)
                return coverset, nlp

            if fval < minsinf:
                winner = i
                minsinf = fval
                holdset = create_candidate_set_func(result, activeA, idx2, k, weights)
                if len(holdset) == 1:
                    nextwinner = holdset

            activeA[i] = True

        if winner is not None:
            coverset.append(winner)
            activeA[winner] = False
            if nextwinner:
                coverset.extend(nextwinner)
                return coverset, nlp

        candidate = holdset.copy()

    return coverset, nlp


def big_m_coverset(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray
) -> Tuple[List[int], Optional[int]]:
    """
    Find a cover set using a Big-M MILP.

    Args:
        A: Constraint matrix.
        b: RHS vector.
        weights: Constraint weights.

    Returns:
        coverset: Selected constraints.
        nlp: None (not applicable to MILP).
    """
    m, n = A.shape
    M = 1e6

    c = np.zeros(n + m)
    c[n:] = weights

    A_constraint = np.hstack([A, -M * np.eye(m)])
    bounds = Bounds(
        lb=np.zeros(n + m),
        ub=np.concatenate([np.full(n, np.inf), np.ones(m)])
    )
    integrality = np.concatenate([np.zeros(n), np.ones(m)])
    constraints = LinearConstraint(A_constraint, ub=b)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    y_values = res.x[n:]
    coverset = np.where(y_values > 0.5)[0]

    return coverset.tolist(), None


def grow(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray
) -> Tuple[List[int], int]:
    """
    Grow method (Bailey & Stuckey): Incrementally build maximal feasible subsystem.

    Args:
        A: Constraint matrix.
        b: RHS vector.
        weights: Constraint weights.

    Returns:
        coverset: Removed constraints.
        nlp: Number of LP checks.
    """
    m, _ = A.shape
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


def compute_baseline(
    batch_size: int,
    c: int,
    v: int,
    method_name: str,
    weights: str = "const"
) -> Tuple[float, float, float, Optional[float]]:
    """
    Evaluate a cover set algorithm across multiple infeasible LPs.

    Args:
        batch_size: Number of problem instances.
        c: Number of constraints per instance.
        v: Number of variables per instance.
        method_name: Algorithm name ("Deletion", "Opt", "Chinneck", "Chinneck-Fast").
        weights: Weight type ("const" or "uniform").

    Returns:
        avg_size: Mean cover set weight.
        std_size: Standard deviation.
        avg_time: Mean runtime.
        avg_nlp: Mean LP count (None for MILP).
    """
    size = []
    times = []
    nlp = []

    if method_name == "Deletion":
        method = grow
    elif method_name == "Opt":
        method = big_m_coverset
    elif method_name == "Chinneck":
        method = lambda A, b, w: generate_cover(A, b, create_candidate_set, k=2, weights=w)
    elif method_name == "Chinneck-Fast":
        method = lambda A, b, w: generate_cover(A, b, create_candidate_set_1, k=2, weights=w)
    else:
        raise ValueError("Invalid method name.")

    for j in tqdm(range(batch_size)):
        c_batch, A_ub_batch, b_ub_batch, constraint_weight_batch = generate_random_infeasible_lp(
            1, v, c, weight=weights, seed=j
        )
        _, A_ub, b_ub, constraint_weight = (
            c_batch[0],
            A_ub_batch[0],
            b_ub_batch[0],
            constraint_weight_batch[0],
        )

        start = time.time()
        coverset, current_nlp = method(A_ub, b_ub, constraint_weight)
        end = time.time()

        size.append(constraint_weight[coverset].sum())
        times.append(end - start)
        nlp.append(current_nlp)

    avg_size = np.mean(size)
    std_size = np.std(size)
    avg_time = np.mean(times)
    avg_nlp = np.mean(nlp) if method_name != "Opt" else None

    return avg_size, std_size, avg_time, avg_nlp
