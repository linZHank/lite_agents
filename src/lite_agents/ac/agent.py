from typing import Optional
import gymnasium as gym
import numpy as np
from lite_agents.ac.components import (
    ACBuffer,
    ExperienceBatch,
    CategoricalActor,
    GaussianActor,
    Critic,
)
from flax import nnx
import optax


@nnx.jit
def make_decision_and_assess(rngs: nnx.Rngs, actor, critic: Critic, obs: np.ndarray):
    pi = actor(obs)  # policy: log(pi(a|s))
    act = pi.sample(seed=rngs)
    val = critic(obs)
    return act.squeeze(), val.squeeze()


@nnx.jit
def objective_fn(actor, experience_batch: ExperienceBatch):
    policies = actor(experience_batch.obs)
    log_pi_as = policies.log_prob(experience_batch.act)
    objective_batch = experience_batch.adv * log_pi_as

    return -objective_batch.mean()


@nnx.jit
def loss_fn(critic: Critic, experience_batch: ExperienceBatch):
    vals = critic(experience_batch.obs)
    v_loss = (vals - experience_batch.ret) ** 2  # TODO: use MSE from a lib

    return v_loss.mean()


@nnx.jit
def update_actor_params(actor, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(objective_fn)
    objective, grads = grad_fn(actor, experience_batch)
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def update_critic_params(critic, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(loss_fn)
    v_loss, grads = grad_fn(critic, experience_batch)
    optimizer.update(grads)  # In-place updates.


def learn(
    env_name: str = "CartPole-v1",
    env_options: Optional[dict] = {"render_mode": "rgb_array"},
    seed: int = 0,
    discount: float = 0.99,
    compromise: float = 0.97,
    max_epochs: int = 64,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-4,
    hidden_sizes: tuple = (64, 64),
    min_epoch_episodes: int = 10,  # minimal episodes per epoch
):
    # SETUP
    env = gym.make(env_name, **env_options)
    rngs = nnx.Rngs(seed)
    buffer = ACBuffer([], [], [], [], [], [])
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
    critic = Critic(rngs, env.observation_space.shape[0], hidden_sizes)
    nnx.display(actor)
    nnx.display(critic)
    actor_optimizer = nnx.Optimizer(actor, optax.adamw(actor_lr))
    critic_optimizer = nnx.Optimizer(critic, optax.adamw(critic_lr))
    learning_journal = {
        "episode_idx": 0,
        "step_idx": 0,
        "episode_len": [0],
        "deposit_return": [0.0],
        "averaged_return": [],
    }
    last_obs, info = env.reset()

    # LOOP
    for e in range(max_epochs):
        for st in range(
            (min_epoch_episodes + 1) * env.spec.max_episode_steps
        ):  # at least 10 finished episodes
            act, last_val = make_decision_and_assess(rngs, actor, critic, last_obs)
            next_obs, rew, term, trunc, info = env.step(np.array(act))
            buffer.store_step(last_obs, act, rew, last_val)
            # TODO: update journal in a util function
            learning_journal["step_idx"] += 1
            learning_journal["episode_len"][-1] += 1
            learning_journal["deposit_return"][-1] += rew
            last_obs = next_obs.copy()
            if term or trunc:
                buffer.wrapup_episode(learning_journal["episode_len"][-1], discount)
                # Episode statistics
                learning_journal["episode_idx"] += 1
                learning_journal["averaged_return"].append(
                    sum(learning_journal["deposit_return"])
                    / len(learning_journal["deposit_return"])
                )
                # TODO: need a logger
                print(
                    f"---\nepisode: {learning_journal['episode_idx']}, length: {learning_journal['episode_len'][-1]}, return: {learning_journal['deposit_return'][-1]}\n---\n"
                )
                # Reset episode
                last_obs, _ = env.reset()
                learning_journal["episode_len"].append(0)
                learning_journal["deposit_return"].append(0.0)
                if st > min_epoch_episodes * env.spec.max_episode_steps:
                    break
        # Epoch statistics
        print(
            f"===\nepoch {e + 1} \n\ttotal steps: {learning_journal['step_idx']}\n\taveraged return: {learning_journal['averaged_return'][-1]}\n==="
        )
        exp_batch = buffer.extract_experience()
        update_params(actor, optimizer, exp_batch)
        buffer = VPGBuffer([], [], [], [])
    # TODO: need a plotter
    import matplotlib.pyplot as plt
    from pathlib import Path

    plt.plot(learning_journal["averaged_return"])
    plt.grid(visible=True, axis="y")
    plt.show()
    # plt.savefig(Path(__file__).parent.joinpath(f"{env_name}.png"))


if __name__ == "__main__":
    # TODO: argparse
    learn(
        env_name="LunarLander-v3",
        env_options={"continuous": True, "render_mode": "rgb_array"},
        hidden_sizes=(128, 128),
        max_epochs=128,
    )
