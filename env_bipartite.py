import gymnasium as gym
from generator_ import generate_random_lp, check_feasibility
import numpy as np


def lp_to_graph(A_ub, b_ub, c):
    # given the coefficients of the LP, we can convert it to a bipartite graph
    # return the constraint features, edge indices, edge features, and variable features
    # use numpy arrays to represent the LP coefficients
    num_constraints, num_variables = A_ub.shape
    constraint_features = b_ub[:, np.newaxis]  # right-hand side of the constraints
    variable_features = c[:, np.newaxis]  # objective function coefficients
    edge_indices = np.stack(
        [
            np.repeat(np.arange(num_constraints), num_variables),
            np.tile(np.arange(num_variables), num_constraints),
        ]
    )
    edge_features = A_ub.flatten()[:, np.newaxis]
    return (
        constraint_features / 100,
        edge_indices,
        edge_features / 100,
        variable_features / 100,
    )


class MAXFSEnv(gym.Env):
    def __init__(self, n_ineq_cons, n_vars):
        self.n_ineq_cons = n_ineq_cons
        self.action_space = gym.spaces.Discrete(n_ineq_cons)
        self.observation_space = gym.spaces.Dict(
            {
                "constraint_features": gym.spaces.Box(
                    low=-1, high=1, shape=(n_ineq_cons, 3), dtype=float
                ),
                "variable_features": gym.spaces.Box(
                    low=-1, high=1, shape=(n_vars, 1), dtype=float
                ),
                "edge_indices": gym.spaces.Box(
                    low=0, high=n_ineq_cons, shape=(2, n_ineq_cons * n_vars), dtype=int
                ),
                "edge_features": gym.spaces.Box(
                    low=-1, high=1, shape=(n_ineq_cons * n_vars, 1), dtype=float
                ),
                "mask": gym.spaces.MultiBinary(n_ineq_cons),
            }
        )

        self.n_vars = n_vars

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.current_set_matrix = np.ones((self.n_ineq_cons, 1))
        (self.c, self.A_ub, self.b_ub) = generate_random_lp(
            1, self.n_vars, self.n_ineq_cons
        )
        self.c = self.c[0]
        self.A_ub = self.A_ub[0]
        self.b_ub = self.b_ub[0]
        # print the inequality constraints in the form a1*x1 + a2*x2 + ... <= b ..

        (
            self.constraint_features,
            self.edge_indices,
            self.edge_features,
            self.variable_features,
        ) = lp_to_graph(self.A_ub, self.b_ub, self.c)
        self.current_infeasible = np.zeros((self.n_ineq_cons, 1))
        current_constraint_features = np.concatenate(
            (
                self.constraint_features,
                self.current_set_matrix,
                self.current_infeasible,
            ),
            axis=-1,
        )
        self.mask = np.zeros(self.n_ineq_cons).astype(int)

        return {
            "constraint_features": current_constraint_features,
            "variable_features": self.variable_features,
            "edge_indices": self.edge_indices,
            "edge_features": self.edge_features,
            "mask": self.mask,
        }, {}

    def step(self, action):
        self.current_step += 1
        self.mask[action] = 1

        current_set = set(np.argwhere(self.current_set_matrix == 1).flatten())

        current_set_without_action = list(current_set.difference({action}))
        feasible = check_feasibility(
            self.c * 0,
            self.A_ub[current_set_without_action, :],
            self.b_ub[current_set_without_action],
        )
        done = feasible or np.all(self.mask == 1)
        if not feasible:
            self.current_set_matrix[action] = 0
            reward = -1
        else:
            reward = 0
            # self.current_infeasible[action] = 1
        current_constraint_features = np.concatenate(
            (
                self.constraint_features,
                self.current_set_matrix,
                self.current_infeasible,
            ),
        )

        return (
            {
                "constraint_features": current_constraint_features,
                "variable_features": self.variable_features,
                "edge_indices": self.edge_indices,
                "edge_features": self.edge_features,
                "mask": self.mask,
            },
            reward,
            done,
            False,
            {},
        )


if __name__ == "__main__":
    from utils import make_parallel_env
    from tqdm import tqdm

    env = MAXFSEnv(15, 5)
    all_r = []
    for _ in tqdm(range(10)):

        obs, info = env.reset()
        # print(
        #     obs["constraint_features"].shape,
        #     obs["variable_features"].shape,
        #     obs["edge_indices"].shape,
        #     obs["edge_features"].shape,
        #     obs["mask"].shape,
        # )
        done = False
        r = 0
        k = 0
        while not done:
            mask = obs["mask"]
            action = np.random.choice(np.argwhere(mask == 0).flatten())

            obs, reward, done, _, _ = env.step(action)
            input()
            r += reward
        all_r.append(r)
    print(np.mean(all_r))
    print(np.std(all_r))

    vec_env = make_parallel_env(
        num_envs=16, base_env=MAXFSEnv, n_ineq_cons=10, n_vars=3
    )
    obs, info = vec_env.reset()

    vec_env.close()
