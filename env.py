import gymnasium as gym
from utils import (
    generate_random_infeasible_lp,
    check_lp_feasibility,
    lp_to_matrix,
    lp_to_graph,
    check_sat_satisfiability,
    generate_random_infeasible_cnf,
    cnf_to_matrix,
)
import numpy as np

BASE_SEED = 10000


class MAXFSEnv(gym.Env):
    def __init__(self, n_cons, n_var, weight="const"):
        self.n_cons = n_cons
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

        self.n_var = n_var
        self.weight = weight

    def reset(self, seed=None, options=None):
        self.current_set_matrix = np.ones((self.n_cons))

        (self.c, self.A_ub, self.b_ub, self.constraint_weight) = (
            generate_random_infeasible_lp(
                1, self.n_var, self.n_cons, weight=self.weight, seed=seed
            )
        )

        self.A_ub = self.A_ub[0]
        self.b_ub = self.b_ub[0]
        self.constraint_weight = self.constraint_weight[0]
        self.mask = np.zeros(self.n_cons).astype(int)
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

    def step(self, action):
        self.mask[action] = 1
        current_set = set(np.argwhere(self.current_set_matrix == 1).flatten())
        current_set_without_action = list(current_set.difference({action}))
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
    def __init__(self, n_cons, n_var, weight="const"):
        self.n_cons = n_cons
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

        self.n_var = n_var
        self.weight = weight

    def reset(self, seed=None, options=None):
        self.current_set_matrix = np.ones((self.n_cons))
        self.cnf_example, self.constraint_weight = generate_random_infeasible_cnf(
            num_clauses=self.n_cons,
            num_variables=self.n_var,
            weight=self.weight,
            seed=seed,
        )
        self.mat = cnf_to_matrix(cnf=self.cnf_example, num_variables=self.n_var)

        self.mask = np.zeros(self.n_cons).astype(int)
        self.edge_features = self.mat[:, :, np.newaxis].copy()

        self.constraint_features = np.zeros((self.n_cons, 2))
        self.constraint_features[:, 0] = self.constraint_weight
        self.constraint_features[:, 1] = self.current_set_matrix
        return {
            "edge_features": self.edge_features,
            "constraint_features": self.constraint_features,
            "mask": self.mask,
        }, {}

    def step(self, action):
        self.mask[action] = 1
        current_set = set(np.argwhere(self.current_set_matrix == 1).flatten())
        current_set_without_action = list(current_set.difference({action}))
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
    from utils import make_parallel_env
    import torch

    # env = MAXSATEnv(50, 20, weight="weight")
    # obs, info = env.reset()
    all_r = []
    # print(
    #     obs["constraint_features"].shape, obs["edge_features"].shape, obs["mask"].shape
    # )
    env = MAXFSEnv(150, 30, weight="weight")
    # obs, info = env2.reset()
    # print(
    #     obs["constraint_features"].shape, obs["edge_features"].shape, obs["mask"].shape
    # )

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
    print(np.mean(all_r))
    print(np.std(all_r))

    vec_env = make_parallel_env(num_envs=20, base_env=MAXFSEnv, n_cons=10, n_var=2)
    obs, info = vec_env.reset()
    matrix = lp_to_matrix(
        constraint_features=torch.from_numpy(obs["constraint_features"]).float(),
        edge_features=torch.from_numpy(obs["edge_features"]).float(),
        mask=torch.from_numpy(obs["mask"]).bool(),
    )
    graph = lp_to_graph(
        constraint_features=torch.from_numpy(obs["constraint_features"]).float(),
        edge_features=torch.from_numpy(obs["edge_features"]).float(),
        mask=torch.from_numpy(obs["mask"]).bool(),
    )
    print(graph)
    vec_env = make_parallel_env(num_envs=20, base_env=MAXSATEnv, n_cons=10, n_var=2)
    obs, info = vec_env.reset()
    matrix = lp_to_matrix(
        constraint_features=torch.from_numpy(obs["constraint_features"]).float(),
        edge_features=torch.from_numpy(obs["edge_features"]).float(),
        mask=torch.from_numpy(obs["mask"]).bool(),
    )
    graph = lp_to_graph(
        constraint_features=torch.from_numpy(obs["constraint_features"]).float(),
        edge_features=torch.from_numpy(obs["edge_features"]).float(),
        mask=torch.from_numpy(obs["mask"]).bool(),
    )
