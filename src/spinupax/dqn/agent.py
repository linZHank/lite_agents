from pathlib import Path, PosixPath
from typing import Optional

import gymnasium as gym
import numpy as np

import jax
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


@jax.vmap
def doubleq_error(data, q_pred, q_next, q_duel):
    q_targ = jax.lax.stop_gradient(
        data.rew + data.disct * q_next[q_duel.argmax(axis=-1)]
    )
    td_error = q_targ - q_pred[data.act]
    return td_error


def loss_fn(critic_online, critic_stable, experience_batch):
    qval_pred = critic_online(experience_batch.lobs)
    qval_next = critic_stable(experience_batch.nobs)
    qval_duel = critic_online(experience_batch.nobs)
    td_error = doubleq_error(
        experience_batch,
        qval_pred,
        qval_next,
        qval_duel,
    )
    loss_value = optax.l2_loss(td_error).mean()
    return loss_value


@nnx.jit
def online_update_fn(critic_online, critic_stable, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss_val, grads = grad_fn(critic_online, critic_stable, experience_batch)
    optimizer.update(grads)
    return loss_val


@nnx.jit
def polyak_update(critic_online, critic_stable):
    _, state_online = nnx.split(critic_online)
    graph_def, state_stable = nnx.split(critic_stable)
    state_update = optax.incremental_update(
        new_tensors=state_online,
        old_tensors=state_stable,
        step_size=0.01,
    )
    critic_stable = nnx.merge(graph_def, state_update)
    return critic_stable


@nnx.jit
def resolve_and_assess(rngs, critic, explore_rate, observation):
    q_value = critic(observation)
    greedy_action = q_value.argmax(axis=-1)
    sampled_action = jax.random.randint(
        key=rngs.params(),
        shape=(),
        minval=0,
        maxval=q_value.size,
    )
    selector = jax.random.uniform(key=rngs.params())
    action = jax.lax.select(
        pred=selector > explore_rate,
        on_true=greedy_action,
        on_false=sampled_action,
    )

    return action, q_value


def learn(
    env_name: str = "CartPole-v1",
    env_options: Optional[dict] = {"render_mode": "rgb_array"},
    seed: int = 25,
    discount: float = 0.99,
    capacity: int = int(1e5),
    max_steps=int(1e4),
    warmup_episodes: int = 10,
    learning_rate: float = 3e-4,
    hidden_sizes: tuple = (64, 64),
    epsilon_decay_episodes: int = 100,
    sample_size: int = 64,
    eval_flag: bool = False,
    ckpt_dir: PosixPath = Path("/tmp/spinupax/dqn/checkpoints/"),
    save_per_episode: int = 100,
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
    for st in range(max_steps):
        # Play a step
        pred_act, q_val = resolve_and_assess(rngs, qnet_online, epsilon, last_obs)
        act = pred_act.squeeze()
        next_obs, rew, term, trunc, info = env.step(np.array(act))
        buffer.store_step(last_obs, act, rew, term, next_obs)
        last_obs = next_obs.copy()
        # Train a step
        if journal_learn["episode_idx"] + 1 > warmup_episodes:
            experience_batch = buffer.extract_experience(rngs, sample_size)
            qloss = online_update_fn(
                qnet_online,
                qnet_stable,
                optimizer,
                experience_batch,
            )
            qnet_stable = polyak_update(qnet_online, qnet_stable)
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
            last_obs, _ = env.reset()
            journal_learn["episode_len"].append(0)
            journal_learn["deposit_return"].append(0.0)
            epsilon = epsilon_schedule(journal_learn["episode_idx"])


if __name__ == "__main__":
    # TODO: argparse
    learn()
