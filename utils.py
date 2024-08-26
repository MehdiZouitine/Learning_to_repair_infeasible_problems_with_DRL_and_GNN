import gymnasium as gym


def create_parallel_env(base_env, **kwargs):
    return base_env(**kwargs)


def make_parallel_env(num_envs, base_env, **kwargs):
    envs = []
    for _ in range(num_envs):
        env = create_parallel_env(base_env, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        envs.append(lambda: env)
    vectorized_env = gym.vector.AsyncVectorEnv(envs)
    return vectorized_env
