# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import torch
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import make_parallel_env
from env import MAXFSEnv, MAXSATEnv
from dka_agent import DBAAgent
from gcnn_agent import BipartiteAgent


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = True
    """if toggled, this experiment will save the model in wandb every K iterations"""
    wandb_project_name: str = "MaxFSRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    """the environment type"""
    # Algorithm specific arguments
    total_timesteps: int = 500000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the id of the environment"""
    n_cons: int = 20
    """the size of the problem"""
    n_var: int = 2
    """the maximum number of generations"""
    weight: str = "const"
    """the weight of the constraints"""
    gnn_architecture: str = "gcnn"
    """the architecture of the GNN"""
    env_type: str = "maxsat"
    """the environment type"""
    eval_freq: int = 20
    """the evaluation frequency of the model"""
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
    ent_coef: float = 0
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
    run_name = f"{args.exp_name}__{args.env_type}__{args.gnn_architecture}__w_{args.weight}__c_{args.n_cons}__v_{args.n_var}__{args.seed}__{int(time.time())}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.env_type == "maxfs":
        base_env = MAXFSEnv
    elif args.env_type == "maxsat":
        base_env = MAXSATEnv
    else:
        raise ValueError("Invalid environment type")

    envs = make_parallel_env(
        num_envs=args.num_envs,
        base_env=base_env,
        n_cons=args.n_cons,
        n_var=args.n_var,
        weight=args.weight,
    )
    if args.gnn_architecture == "dka":
        agent = DBAAgent(
            input_dim=envs.single_observation_space["edge_features"].shape[-1]
            + envs.single_observation_space["constraint_features"].shape[-1],
            hidden_dim=128,
            ff_dim=512,
            k_dim=64,
            v_dim=64,
            n_head=3,
            n_layers=4,
            device=device,
        )
    else:
        agent = BipartiteAgent(
            cons_nfeats=envs.single_observation_space["constraint_features"].shape[-1],
            edge_nfeats=envs.single_observation_space["edge_features"].shape[-1],
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
    next_obs_edge_features = torch.Tensor(next_obs["edge_features"]).to(device)
    next_obs_mask = torch.Tensor(next_obs["mask"]).to(device)
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
            obs_edge_features[step] = next_obs_edge_features
            obs_mask[step] = next_obs_mask
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs_constraint_features, next_obs_edge_features, next_obs_mask
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
            next_obs_edge_features = torch.Tensor(next_obs["edge_features"]).to(device)
            next_obs_mask = torch.Tensor(next_obs["mask"]).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        all_returns.append(info["episode"]["r"])
                        all_lengths.append(info["episode"]["l"])

        writer.add_scalar("charts/episodic_return", np.mean(all_returns), global_step)
        writer.add_scalar("charts/episodic_length", np.mean(all_lengths), global_step)
        print(
            f"[{global_step}] Avg Return: {np.mean(all_returns)} / Avg Length: {np.mean(all_lengths)} "
        )
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs_constraint_features, next_obs_edge_features, next_obs_mask
            ).reshape(1, -1)
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs_constraint_features[mb_inds],
                    b_obs_edge_features[mb_inds],
                    b_obs_mask[mb_inds],
                    b_actions[mb_inds],
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
        if iteration % args.eval_freq == 0:
            # eval the agent in on size (n,m) = (2,10), (5,20), (10,50), (20,100), (30,150)
            # The number of evaluations per size depend on the problem size (1000 for (2,10),(5,20),and (10,50), 100 for (15,10) and (30,150))
            # for each size measure the average return and the average size of the max feasible set and the time taken to solve the problem
            # we will use the same seed for all the evaluations from 0 to 1000 for the first three sizes and from 0 to 100 for the last two sizes

            if args.env_type == "maxfs":
                env_size = [(2, 10), (5, 20), (10, 50), (20, 100), (30, 150)]
                eval_envs_list = [
                    MAXFSEnv(n_var=n, n_cons=m, weight=args.weight) for n, m in env_size
                ]
            elif args.env_type == "maxsat":
                env_size = [(2, 20), (4, 40), (3, 60), (5, 100), (10, 200)]
                eval_envs_list = [
                    MAXSATEnv(n_var=n, n_cons=m, weight=args.weight)
                    for n, m in env_size
                ]
            else:
                raise ValueError("Invalid environment type")
            agent.eval()
            num_evals = 300
            with torch.no_grad():
                print("Evaluating ...")
                for idx, eval_env in tqdm(enumerate(eval_envs_list)):
                    if args.env_type == "maxfs":
                        if idx < 3:
                            num_evals = 300
                        else:
                            num_evals = 50
                    start = time.time()
                    cost = []
                    coverset_size = []
                    for k in range(num_evals):
                        obs, _ = eval_env.reset(seed=k)
                        done = False
                        length = 0
                        total_rewards = 0
                        while not done:
                            logits = agent(
                                torch.Tensor(obs["constraint_features"])
                                .unsqueeze(0)
                                .to(device),
                                torch.Tensor(obs["edge_features"])
                                .unsqueeze(0)
                                .to(device),
                                torch.Tensor(obs["mask"]).unsqueeze(0).to(device),
                            )
                            action = torch.argmax(logits, dim=1)
                            obs, reward, done, _, _ = eval_env.step(action.cpu().item())
                            length += 1
                            total_rewards += reward

                        cost.append(total_rewards)
                        coverset_size.append(length)
                    end = time.time()
                    total_time = (end - start) / num_evals
                    writer.add_scalar(
                        f"charts_eval/{env_size[idx]}_time", total_time, global_step
                    )
                    writer.add_scalar(
                        f"charts_eval/{env_size[idx]}_mean_cost",
                        np.mean(cost),
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts_eval/{env_size[idx]}_std_cost",
                        np.std(cost),
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts_eval/{env_size[idx]}_mean_coverset_size",
                        np.mean(coverset_size),
                        global_step,
                    )
                    writer.add_scalar(
                        f"charts_eval/{env_size[idx]}_std_coverset_size",
                        np.std(coverset_size),
                        global_step,
                    )
                    print(
                        f"Evalutation results for {env_size[idx]} : Time {total_time}, Mean Cost {np.mean(cost)}, Mean Cover Set Size {np.mean(coverset_size)}"
                    )
                    # save the models in the folder models only when env_size[idx] == (args.n, args.m)
                    if args.save_model:
                        if env_size[idx] == (args.n_var, args.n_cons):
                            if args.track:
                                model_path = f"models/best_model_{wandb.run.id}_{args.n_var}_{args.n_cons}_{global_step}_{np.mean(cost)}.pth"
                                wandb.save(model_path)
                                torch.save(agent.state_dict(), model_path)
                            else:
                                model_path = f"models/best_model_{args.n_var}_{args.n_cons}_{global_step}_{np.mean(cost)}.pth"
                                torch.save(agent.state_dict(), model_path)

            agent.train()

    envs.close()
    writer.close()
