from typing import Optional
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from lite_agents.ac.components import (
    ACBuffer,
    ExperienceBatch,
    CategoricalActor,
    GaussianActor,
    Critic,
)
from flax import nnx
import optax

import matplotlib.pyplot as plt
from pathlib import Path


@nnx.jit
def objective_fn(actor, experience_batch: ExperienceBatch):
    policies = actor(experience_batch.obs)
    log_pi_as = policies.log_prob(experience_batch.act)
    objective_batch = experience_batch.adv * log_pi_as

    return -objective_batch.mean()  # A(s_t, a_t) log(pi(a_t|s_t))


@nnx.jit
def loss_fn(critic: Critic, experience_batch: ExperienceBatch):
    pred_vals = critic(experience_batch.obs)
    targ_vals = experience_batch.ret
    mse_loss = (pred_vals - targ_vals) ** 2  # TODO: use MSE from a lib

    return mse_loss.mean()


@nnx.jit
def update_actor_params(actor, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(objective_fn)
    objective, grads = grad_fn(actor, experience_batch)
    optimizer.update(grads)  # In-place updates.

    return objective


@nnx.jit
def update_critic_params(critic, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(loss_fn)
    v_loss, grads = grad_fn(critic, experience_batch)
    optimizer.update(grads)  # In-place updates.

    return v_loss


@nnx.jit
def resolve_and_assess(rngs: nnx.Rngs, actor, critic: Critic, obs: np.ndarray):
    pi = actor(jnp.expand_dims(obs, axis=0))
    act = pi.sample(seed=rngs)
    val = critic(obs)

    return act.squeeze(axis=-1), val.squeeze()


def learn(
    env_name: str = "CartPole-v1",
    env_options: Optional[dict] = {"render_mode": "rgb_array"},
    seed: int = 25,
    discount: float = 0.99,
    tradeoff: float = 0.97,
    max_epochs: int = 64,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-4,
    critic_update_iters=80,
    hidden_sizes: tuple = (64, 64),
    min_epoch_episodes: int = 5,  # minimal episodes per epoch
    eval_flag: bool = False,
):
    # SETUP
    env = gym.make(env_name, **env_options)
    rngs = nnx.Rngs(seed)
    if isinstance(env.action_space, gym.spaces.Box):
        actor = GaussianActor(
            rngs,
            env.observation_space.shape[0],
            env.action_space.shape[0],
            hidden_sizes,
        )
    elif isinstance(env.action_space, gym.spaces.Discrete):
        actor = CategoricalActor(
            rngs,
            env.observation_space.shape[0],
            env.action_space.n,
            hidden_sizes,
        )
    critic = Critic(
        rngs=rngs,
        observation_dims=env.observation_space.shape[0],
        hidden_sizes=hidden_sizes,
    )
    nnx.display(actor)
    nnx.display(critic)
    actor_optimizer = nnx.Optimizer(actor, optax.adamw(actor_lr))
    critic_optimizer = nnx.Optimizer(critic, optax.adamw(critic_lr))
    buffer = ACBuffer([], [], [], [], [], [])
    journal_learn = {
        "episode_idx": 0,
        "step_idx": 0,
        "episode_len": [0],
        "deposit_return": [0.0],
        "averaged_return": [],
    }

    # LOOP
    last_obs, info = env.reset()
    for e in range(max_epochs):
        for st in range((min_epoch_episodes + 1) * env.spec.max_episode_steps):
            # Play a step
            act, last_val = resolve_and_assess(rngs, actor, critic, last_obs)
            next_obs, rew, term, trunc, info = env.step(np.array(act.squeeze()))
            buffer.store_step(last_obs, act, rew, last_val)
            journal_learn["step_idx"] += 1  # TODO: update journal in a util function
            journal_learn["episode_len"][-1] += 1
            journal_learn["deposit_return"][-1] += rew
            last_obs = next_obs.copy()
            # Wrap up episode
            if term or trunc:
                if trunc:
                    eoe_val = jnp.squeeze(critic(last_obs))
                else:
                    eoe_val = jnp.zeros(shape=())
                buffer.wrapup_episode(
                    eoe_val, journal_learn["episode_len"][-1], discount, tradeoff
                )
                journal_learn["episode_idx"] += 1
                journal_learn["averaged_return"].append(
                    sum(journal_learn["deposit_return"])
                    / len(journal_learn["deposit_return"])
                )
                # TODO: need a logger
                print(
                    f"---\nepisode: {journal_learn['episode_idx']}, length: {journal_learn['episode_len'][-1]}, return: {journal_learn['deposit_return'][-1]}\n---\n"
                )
                # Reset episode
                last_obs, info = env.reset()
                journal_learn["episode_len"].append(0)
                journal_learn["deposit_return"].append(0.0)
                if st > min_epoch_episodes * env.spec.max_episode_steps:
                    break
        # Wrap up epoch
        print(
            f"===\nepoch {e + 1} \n\ttotal steps: {journal_learn['step_idx']}\n\taveraged return: {journal_learn['averaged_return'][-1]}\n==="
        )
        experience_batch = buffer.extract_experience()
        update_actor_params(actor, actor_optimizer, experience_batch)
        for _ in range(critic_update_iters):
            update_critic_params(critic, critic_optimizer, experience_batch)
        buffer = ACBuffer([], [], [], [], [], [])
    # TODO: need a plotter
    plt.plot(journal_learn["averaged_return"])
    plt.grid(visible=True, axis="y")
    plt.show()
    # plt.savefig(Path(__file__).parent.joinpath(f"{env_name}.png"))

    # Evaluation
    if eval_flag:  # TODO: save and load
        env_options["render_mode"] = "human"
        env = gym.make(env_name, **env_options)
        obs, _ = env.reset()
        episode_return = 0.0
        for _ in range(env.spec.max_episode_steps):
            act, _ = resolve_and_assess(rngs, actor, critic, obs)
            obs, rew, term, trunc, _ = env.step(np.array(act.squeeze()))
            episode_return += rew
            if term or trunc:
                print(f"\n---return: {episode_return}---\n")
                break


if __name__ == "__main__":
    # TODO: argparse
    learn(
        # env_name="CartPole-v1",
        env_name="LunarLander-v3",
        env_options={"continuous": True, "render_mode": "rgb_array"},
        hidden_sizes=(128, 128),
        max_epochs=64,
        critic_update_iters=50,
        eval_flag=True,
    )
