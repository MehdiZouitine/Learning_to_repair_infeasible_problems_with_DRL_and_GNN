import gymnasium as gym
from utils import (
    generate_random_infeasible_lp,
    check_lp_feasibility,
    check_sat_satisfiability,
    generate_random_infeasible_cnf,
    cnf_to_matrix,
)
import numpy as np
from typing import Optional, Tuple, Dict

BASE_SEED = 10000


class MAXFSEnv(gym.Env):
    """
    MAXFS Environment (Maximum Feasible Subsystem) for LP problems.

    Observation space:
        - edge_features: [n_cons, n_var, 1]
        - constraint_features: [n_cons, 3] -> [weight, RHS scaled, active]
        - mask: binary mask for chosen constraints

    Action space:
        - Discrete selection of constraints to remove.
    """

    def __init__(self, n_cons: int, n_var: int, weight: str = "const"):
        self.n_cons = n_cons
        self.n_var = n_var
        self.weight = weight

        self.action_space = gym.spaces.Discrete(n_cons)
        self.observation_space = gym.spaces.Dict(
            {
                "edge_features": gym.spaces.Box(
                    low=-1, high=1, shape=(n_cons, n_var, 1), dtype=float
                ),
                "constraint_features": gym.spaces.Box(
                    low=-1, high=1, shape=(n_cons, 3), dtype=float
                ),
                "mask": gym.spaces.MultiBinary(n_cons),
            }
        )

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[Dict, Dict]:
        """
        Reset environment with new random infeasible LP.

        Returns:
            observation, info dict (empty).
        """
        self.current_set_matrix = np.ones((self.n_cons))

        self.c, self.A_ub, self.b_ub, self.constraint_weight = generate_random_infeasible_lp(
            1, self.n_var, self.n_cons, weight=self.weight, seed=seed
        )
        self.A_ub = self.A_ub[0]
        self.b_ub = self.b_ub[0]
        self.constraint_weight = self.constraint_weight[0]

        self.mask = np.zeros(self.n_cons, dtype=int)

        self.edge_features = self.A_ub[:, :, np.newaxis].copy() / 100

        self.constraint_features = np.zeros((self.n_cons, 3))
        self.constraint_features[:, 0] = self.constraint_weight
        self.constraint_features[:, 1] = self.b_ub / 100
        self.constraint_features[:, 2] = self.current_set_matrix

        return {
            "edge_features": self.edge_features,
            "constraint_features": self.constraint_features,
            "mask": self.mask,
        }, {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Remove a constraint and check feasibility.

        Reward is the negative weight of the removed constraint.

        Returns:
            observation, reward, done, truncated, info
        """
        self.mask[action] = 1

        current_set = set(np.argwhere(self.current_set_matrix == 1).flatten())
        current_set_without_action = list(current_set - {action})

        feasible = check_lp_feasibility(
            self.A_ub[current_set_without_action, :],
            self.b_ub[current_set_without_action],
        )
        done = feasible or np.all(self.mask == 1)
        reward = -self.constraint_weight[action]

        if not feasible:
            self.current_set_matrix[action] = 0

        self.constraint_features[:, 2] = self.current_set_matrix

        return (
            {
                "edge_features": self.edge_features,
                "constraint_features": self.constraint_features,
                "mask": self.mask,
            },
            reward,
            done,
            False,
            {},
        )


class MAXSATEnv(gym.Env):
    """
    MAXSAT Environment for CNF SAT problems.

    Observation space:
        - edge_features: [n_cons, n_var, 1]
        - constraint_features: [n_cons, 2] -> [weight, active]
        - mask: binary mask for chosen clauses

    Action space:
        - Discrete selection of clauses to remove.
    """

    def __init__(self, n_cons: int, n_var: int, k_max: int, weight: str = "const"):
        self.n_cons = n_cons
        self.n_var = n_var
        self.k_max = k_max
        self.weight = weight

        self.action_space = gym.spaces.Discrete(n_cons)
        self.observation_space = gym.spaces.Dict(
            {
                "edge_features": gym.spaces.Box(
                    low=-1, high=1, shape=(n_cons, n_var, 1), dtype=float
                ),
                "constraint_features": gym.spaces.Box(
                    low=-1, high=1, shape=(n_cons, 2), dtype=float
                ),
                "mask": gym.spaces.MultiBinary(n_cons),
            }
        )

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[Dict, Dict]:
        """
        Reset environment with a new random infeasible CNF.

        Returns:
            observation, info dict (empty).
        """
        self.current_set_matrix = np.ones((self.n_cons))

        self.cnf_example, self.constraint_weight = generate_random_infeasible_cnf(
            num_clauses=self.n_cons,
            num_variables=self.n_var,
            weight=self.weight,
            k_max=self.k_max,
            seed=seed,
        )

        self.mat = cnf_to_matrix(cnf=self.cnf_example, num_variables=self.n_var)

        self.mask = np.zeros(self.n_cons, dtype=int)

        self.edge_features = self.mat[:, :, np.newaxis].copy()

        self.constraint_features = np.zeros((self.n_cons, 2))
        self.constraint_features[:, 0] = self.constraint_weight
        self.constraint_features[:, 1] = self.current_set_matrix

        return {
            "edge_features": self.edge_features,
            "constraint_features": self.constraint_features,
            "mask": self.mask,
        }, {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Remove a clause and check satisfiability.

        Reward is the negative weight of the removed clause.

        Returns:
            observation, reward, done, truncated, info
        """
        self.mask[action] = 1

        current_set = set(np.argwhere(self.current_set_matrix == 1).flatten())
        current_set_without_action = list(current_set - {action})

        feasible = check_sat_satisfiability(
            [self.cnf_example[i] for i in current_set_without_action]
        )
        done = feasible or np.all(self.mask == 1)
        reward = -self.constraint_weight[action]

        if not feasible:
            self.current_set_matrix[action] = 0

        self.constraint_features[:, 1] = self.current_set_matrix

        return (
            {
                "edge_features": self.edge_features,
                "constraint_features": self.constraint_features,
                "mask": self.mask,
            },
            reward,
            done,
            False,
            {},
        )


if __name__ == "__main__":
    from tqdm import tqdm

    all_r = []

    env = MAXSATEnv(15, 3, k_max=3, weight="const")

    for _ in tqdm(range(100)):
        obs, info = env.reset()
        done = False
        r = 0
        k = 0
        while not done:
            mask = obs["mask"]
            action = np.random.choice(np.argwhere(mask == 0).flatten())
            obs, reward, done, _, _ = env.step(action)
            r += reward
            k += 1
        all_r.append(r)

    print(f"Mean episodic reward: {np.mean(all_r)}")
    print(f"Std episodic reward: {np.std(all_r)}")
