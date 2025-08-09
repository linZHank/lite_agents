from pathlib import Path, PosixPath
from typing import Optional

import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from spinupax.dqn.components import (
    DQNBuffer,
    ExperienceBatch,
    QValueNet,
)

import matplotlib.pyplot as plt


def learn(
    env_name: str = "CartPole-v1",
    env_options: Optional[dict] = {"render_mode": "rgb_array"},
    seed: int = 25,
    discount: float = 0.99,
    capacity: int = int(1e5),
    max_steps=int(1e5),
    warmup_episodes: int = 10,
    learning_rate: float = 3e-4,
    hidden_sizes: tuple = (64, 64),
    epsilon_decay_episodes: int = 100,
    experience_sample_size: int = 512,
    eval_flag: bool = False,
    ckpt_dir: PosixPath = Path("/tmp/spinupax/cartpole/dqn/checkpoints/"),
    save_per_epoch: int = 10,
):
    # SETUP
    env = gym.make(env_name, **env_options)
    rngs = nnx.Rngs(seed)
    qnet_online = QValueNet(
        rngs=rngs,
        observation_dims=env.observation_space.shape[0],
        action_dims=env.action_space.n,
        hidden_sizes=hidden_sizes,
    )
    qnet_stable = QValueNet(
        rngs=rngs,
        observation_dims=env.observation_space.shape[0],
        action_dims=env.action_space.n,
        hidden_sizes=hidden_sizes,
    )
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.05,
        transition_steps=epsilon_decay_episodes,
        transition_begin=warmup_episodes,
    )
    optimizer = nnx.Optimizer(qnet_online, optax.adamw(learning_rate=learning_rate))
    buffer = DQNBuffer(
        observation_dims=env.observation_space.shape[0],
        discount=discount,
        capacity=capacity,
    )
    journal_learn = {
        "episode_idx": 0,
        "step_idx": 0,
        "episode_len": [0],
        "deposit_return": [0.0],
        "averaged_return": [],
    }

    # LOOP
    epsilon = epsilon_schedule(0)
    last_obs, info = env.reset()
    for st in range(100 * env.spec.max_episode_steps):
        # Play a step
        pred_act, q_val = resolve_and_assess(rngs, qnet_online, epsilon, last_obs)
        act = pred_act.squeeze()
        next_obs, rew, term, trunc, info = env.step(np.array(act))
        buffer.store_step(last_obs, act, rew, term, next_obs)
        last_obs = next_obs.copy()
        if journal_learn["episode_idx"] + 1 > warmup_episodes:
            experience_batch = buffer.extract_experience(rngs, 512)
            qloss = online_update_fn(
                qnet_online, qnet_stable, optimizer, experience_batch
            )
            qnet_stable = polyak_update(qnet_online, qnet_stable)
            print(f"q value loss: {qloss}")
        # Step statistics
        journal_learn["step_idx"] += 1  # TODO: update journal in a util function
        journal_learn["episode_len"][-1] += 1
        journal_learn["deposit_return"][-1] += rew
        if term or trunc:
            journal_learn["episode_idx"] += 1
            journal_learn["averaged_return"].append(
                sum(journal_learn["deposit_return"])
                / len(journal_learn["deposit_return"])
            )
            # TODO: need a logger
            print(
                f"---\nepisode: {journal_learn['episode_idx']}, epsilon: {epsilon}, length: {journal_learn['episode_len'][-1]}, return: {journal_learn['deposit_return'][-1]}\n---\n"
            )
            ep_return = 0
            pobs, _ = env.reset()
            journal_learn["episode_len"].append(0)
            journal_learn["deposit_return"].append(0.0)
            epsilon = epsilon_schedule(journal_learn["episode_idx"])

        env = gym.make(env_name, **env_options)
        rngs = nnx.Rngs(seed)
