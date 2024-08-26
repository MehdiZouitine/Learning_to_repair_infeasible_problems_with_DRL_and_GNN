import gymnasium as gym
from generator_ import (
    generate_random_lp,
    lp_to_matrix,
    check_feasibility,
    generate_random_ilp,
    generate_random_milp,
)
import numpy as np


class MAXFSEnv(gym.Env):
    def __init__(self, n_ineq_cons, n_vars, problem_type="LP"):
        self.n_ineq_cons = n_ineq_cons
        self.action_space = gym.spaces.Discrete(n_ineq_cons)
        self.observation_space = gym.spaces.Dict(
            {
                "matrix": gym.spaces.Box(
                    low=-1, high=1, shape=(n_ineq_cons, n_vars, 7), dtype=float
                ),
                "mask": gym.spaces.MultiBinary(n_ineq_cons),
            }
        )

        self.n_vars = n_vars
        self.problem_type = problem_type

    def reset(self, seed=None, options=None):
        self.current_set_matrix = np.ones((self.n_ineq_cons))
        if self.problem_type == "LP":
            (self.c, self.A_ub, self.b_ub, self.integrality) = generate_random_lp(
                1, self.n_vars, self.n_ineq_cons, seed=seed
            )
        elif self.problem_type == "ILP":
            (self.c, self.A_ub, self.b_ub, self.integrality) = generate_random_ilp(
                1, self.n_vars, self.n_ineq_cons, seed=seed
            )
        elif self.problem_type == "MILP":
            (self.c, self.A_ub, self.b_ub, self.integrality) = generate_random_milp(
                1, self.n_vars, self.n_ineq_cons, seed=seed
            )
        self.instance = lp_to_matrix(
            self.A_ub, self.b_ub, self.c * 0, self.integrality
        )[0]
        self.c = self.c[0]
        self.A_ub = self.A_ub[0]
        self.b_ub = self.b_ub[0]
        self.mask = np.zeros(self.n_ineq_cons).astype(int)

        repeated_current_set_matrix = np.repeat(
            self.current_set_matrix[:, np.newaxis], self.n_vars, axis=1
        )[:, :, np.newaxis]
        obs = np.concatenate(
            (self.instance.copy(), repeated_current_set_matrix),
            axis=-1,
        )
        self.current_infeasible = np.zeros(self.n_ineq_cons)
        repeated_current_infeasible = np.repeat(
            self.current_infeasible[:, np.newaxis], self.n_vars, axis=1
        )[:, :, np.newaxis]
        obs = np.concatenate((obs, repeated_current_infeasible), axis=-1)
        return {"matrix": obs, "mask": self.mask}, {}

    def step(self, action):
        self.mask[action] = 1
        current_set = set(np.argwhere(self.current_set_matrix == 1).flatten())
        current_set_without_action = list(current_set.difference({action}))
        feasible = check_feasibility(
            self.c * 0,
            self.A_ub[current_set_without_action, :],
            self.b_ub[current_set_without_action],
            integrality=self.integrality,
        )
        done = feasible or np.all(self.mask == 1)
        if not feasible:
            self.current_set_matrix[action] = 0
            reward = -1
        else:
            reward = 0
        repeated_current_set_matrix = np.repeat(
            self.current_set_matrix[:, np.newaxis], self.n_vars, axis=1
        )[:, :, np.newaxis]
        obs = np.concatenate(
            (self.instance.copy(), repeated_current_set_matrix),
            axis=-1,
        )
        repeated_current_infeasible = np.repeat(
            self.current_infeasible[:, np.newaxis], self.n_vars, axis=1
        )[:, :, np.newaxis]
        obs = np.concatenate((obs, repeated_current_infeasible), axis=-1)
        return {"matrix": obs, "mask": self.mask}, reward, done, False, {}


if __name__ == "__main__":
    from tqdm import tqdm

    env = MAXFSEnv(15, 2)
    all_r = []
    for _ in tqdm(range(100)):

        obs, info = env.reset()

        done = False
        r = 0
        while not done:
            mask = obs["mask"]
            action = np.random.choice(np.argwhere(mask == 0).flatten())
            obs, reward, done, _, _ = env.step(action)
            r += reward
        all_r.append(r)
    print(np.mean(all_r))
    print(np.std(all_r))


# vec_env = make_parallel_env(20, 100, 20)
#     # obs, info = vec_env.reset()
#     # vec_env.step(np.zeros(20).astype(int))
