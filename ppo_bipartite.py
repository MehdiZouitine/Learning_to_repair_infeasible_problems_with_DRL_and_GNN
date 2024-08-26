# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import make_parallel_env
from env import MAXFSEnv
from bipartite import BipartiteAgent, graph_to_pytorch_geometric_data
from env_bipartite import MAXFSEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MaxFSRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 500000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the id of the environment"""
    n_ineq_cons: int = 40
    """the size of the problem"""
    n_var: int = 5
    """the maximum number of generations"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


if __name__ == "__main__":

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # # TRY NOT TO MODIFY: seeding
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_parallel_env(
        num_envs=args.num_envs,
        base_env=MAXFSEnv,
        n_ineq_cons=args.n_ineq_cons,
        n_vars=args.n_var,
    )

    agent = BipartiteAgent(
        cons_nfeats=3,
        edge_nfeats=1,
        var_nfeats=1,
        emb_size=128,
        device=device,
        n_layers=4,
    )

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    agent = agent.to(device)
    # ALGO Logic: Storage setup
    obs_constraint_features = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_observation_space["constraint_features"].shape
    ).to(device)
    obs_variable_features = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_observation_space["variable_features"].shape
    ).to(device)
    obs_edge_indices = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_observation_space["edge_indices"].shape
    ).to(device)
    obs_edge_features = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_observation_space["edge_features"].shape
    ).to(device)
    obs_mask = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space["mask"].shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_constraint_features = torch.Tensor(next_obs["constraint_features"]).to(
        device
    )
    next_obs_variable_features = torch.Tensor(next_obs["variable_features"]).to(device)
    next_obs_edge_indices = torch.Tensor(next_obs["edge_indices"]).to(device)
    next_obs_edge_features = torch.Tensor(next_obs["edge_features"]).to(device)
    next_obs_mask = torch.Tensor(next_obs["mask"]).to(device).bool()
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        all_returns = []
        all_lengths = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_constraint_features[step] = next_obs_constraint_features
            obs_variable_features[step] = next_obs_variable_features
            obs_edge_indices[step] = next_obs_edge_indices
            obs_edge_features[step] = next_obs_edge_features
            obs_mask[step] = next_obs_mask
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                bipartite_graph = graph_to_pytorch_geometric_data(
                    next_obs_constraint_features,
                    next_obs_edge_indices,
                    next_obs_edge_features,
                    next_obs_variable_features,
                )
                action, logprob, _, value = agent.get_action_and_value(
                    left_features=bipartite_graph.constraint_features,
                    edge_indices=bipartite_graph.edge_index.long(),
                    edge_features=bipartite_graph.edge_attr,
                    right_features=bipartite_graph.variable_features,
                    mask=next_obs_mask,
                )

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_constraint_features = torch.Tensor(
                next_obs["constraint_features"]
            ).to(device)
            next_obs_variable_features = torch.Tensor(next_obs["variable_features"]).to(
                device
            )
            next_obs_edge_indices = torch.Tensor(next_obs["edge_indices"]).to(device)
            next_obs_edge_features = torch.Tensor(next_obs["edge_features"]).to(device)
            next_obs_mask = torch.Tensor(next_obs["mask"]).to(device).bool()
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        all_returns.append(info["episode"]["r"])
                        all_lengths.append(info["episode"]["l"])
        if iteration % 1 == 0:
            writer.add_scalar(
                "charts/episodic_return", np.mean(all_returns), global_step
            )
            writer.add_scalar(
                "charts/episodic_length", np.mean(all_lengths), global_step
            )
            print(
                f"[{global_step}] Avg Return: {np.mean(all_returns)} / Avg Length: {np.mean(all_lengths)} / Set size {args.n_ineq_cons - np.mean(all_lengths)}"
            )

        # bootstrap value if not done
        with torch.no_grad():
            bipartite_graph = graph_to_pytorch_geometric_data(
                constraint_features=next_obs_constraint_features,
                edge_indices=next_obs_edge_indices,
                edge_features=next_obs_edge_features,
                variable_features=next_obs_variable_features,
            )
            next_value = agent.get_value(
                left_features=bipartite_graph.constraint_features,
                edge_indices=bipartite_graph.edge_index.long(),
                edge_features=bipartite_graph.edge_attr,
                right_features=bipartite_graph.variable_features,
                mask=next_obs_mask,
            ).flatten()
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs_constraint_features = obs_constraint_features.reshape(
            (-1,) + envs.single_observation_space["constraint_features"].shape
        )
        b_obs_variable_features = obs_variable_features.reshape(
            (-1,) + envs.single_observation_space["variable_features"].shape
        )
        b_obs_edge_indices = obs_edge_indices.reshape(
            (-1,) + envs.single_observation_space["edge_indices"].shape
        )
        b_obs_edge_features = obs_edge_features.reshape(
            (-1,) + envs.single_observation_space["edge_features"].shape
        )
        b_obs_mask = obs_mask.reshape(
            (-1,) + envs.single_observation_space["mask"].shape
        )
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                bipartite_graph = graph_to_pytorch_geometric_data(
                    constraint_features=b_obs_constraint_features[mb_inds],
                    edge_indices=b_obs_edge_indices[mb_inds],
                    edge_features=b_obs_edge_features[mb_inds],
                    variable_features=b_obs_variable_features[mb_inds],
                )

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    left_features=bipartite_graph.constraint_features,
                    edge_indices=bipartite_graph.edge_index.long(),
                    edge_features=bipartite_graph.edge_attr,
                    right_features=bipartite_graph.variable_features,
                    action=b_actions[mb_inds],
                    mask=b_obs_mask[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if iteration % 1 == 0:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

    envs.close()
    writer.close()
