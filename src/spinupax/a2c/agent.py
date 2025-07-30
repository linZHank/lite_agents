from pathlib import Path, PosixPath
from typing import Optional

import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from spinupax.a2c.components import (
    A2CBuffer,
    ExperienceBatch,
    CategoricalPolicyNet,
    GaussianPolicyNet,
    ValueNet,
)

import matplotlib.pyplot as plt


@nnx.jit
def objective_fn(actor, experience_batch: ExperienceBatch):
    policies = actor(experience_batch.obs)
    log_pi_as = policies.log_prob(experience_batch.act)
    objective_batch = experience_batch.adv * log_pi_as

    return -objective_batch.mean()  # A(s_t, a_t) log(pi(a_t|s_t))


@nnx.jit
def loss_fn(critic: ValueNet, experience_batch: ExperienceBatch):
    pred_vals = critic(experience_batch.obs)
    targ_vals = experience_batch.ret
    l2_loss = optax.l2_loss(pred_vals, targ_vals)

    return l2_loss.mean()


@nnx.jit
def update_actor_params(
    actor, actor_optimizer: nnx.Optimizer, experience_batch: ExperienceBatch
):
    grad_fn = nnx.value_and_grad(objective_fn)
    objective, grads = grad_fn(actor, experience_batch)
    actor_optimizer.update(grads)  # In-place updates.

    return objective


@nnx.jit
def update_critic_params(
    critic: ValueNet, critic_optimizer: nnx.Optimizer, experience_batch: ExperienceBatch
):
    grad_fn = nnx.value_and_grad(loss_fn)
    v_loss, grads = grad_fn(critic, experience_batch)
    critic_optimizer.update(grads)  # In-place updates.

    return v_loss


@nnx.jit
def resolve_and_assess(
    rngs: nnx.Rngs,
    actor: nnx.Module,
    critic: ValueNet,
    obs: np.ndarray,
):
    pi = actor(jnp.expand_dims(obs, axis=0))
    act = pi.sample(seed=rngs)
    val = critic(obs)

    return act.squeeze(axis=0), val.squeeze()


def save_models(
    ckpt_dir: PosixPath,
    epoch_idx: int,
    actor: nnx.Module,
    critic: ValueNet,
    checkpointer: ocp.StandardCheckpointer,
):
    _, actor_state = nnx.split(actor)
    _, critic_state = nnx.split(critic)
    actor_path = ckpt_dir / f"actor/state_{epoch_idx}"
    critic_path = ckpt_dir / f"critic/state_{epoch_idx}"
    checkpointer.save(actor_path, actor_state)
    checkpointer.save(critic_path, critic_state)
    print(f"Actor state saved at: {actor_path}")
    print(f"Critic state saved at: {critic_path}")


def load_models(
    ckpt_dir: PosixPath,
    epoch_idx: int,
    actor_graphdef,
    critic_graphdef,
    abstract_actor_state,
    abstract_critic_state,
):
    actor_path = ckpt_dir / f"actor/state_{epoch_idx}"
    critic_path = ckpt_dir / f"critic/state_{epoch_idx}"
    checkpointer = ocp.StandardCheckpointer()
    restored_actor_state = checkpointer.restore(actor_path, abstract_actor_state)
    restored_critic_state = checkpointer.restore(critic_path, abstract_critic_state)
    print("Actor State restored: ")
    nnx.display(restored_actor_state)
    print("Critic State restored: ")
    nnx.display(restored_critic_state)
    restored_actor = nnx.merge(actor_graphdef, restored_actor_state)
    restored_critic = nnx.merge(critic_graphdef, restored_actor_state)

    return restored_actor, restored_critic


def learn(
    env_name: str = "CartPole-v1",
    env_options: Optional[dict] = {"render_mode": "rgb_array"},
    seed: int = 25,
    discount: float = 0.99,
    tradeoff: float = 0.97,
    max_epochs: int = 32,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-4,
    critic_update_iters=80,
    hidden_sizes: tuple = (64, 64),
    min_epoch_episodes: int = 5,  # minimal episodes per epoch
    eval_flag: bool = False,
    ckpt_dir: PosixPath = Path("/tmp/spinupax/a2c/checkpoints/"),
    save_per_epoch: int = 10,
):
    # SETUP
    env = gym.make(env_name, **env_options)
    rngs = nnx.Rngs(seed)
    if isinstance(env.action_space, gym.spaces.Box):
        actor = GaussianPolicyNet(
            rngs=rngs,
            observation_dims=env.observation_space.shape[0],
            action_dims=env.action_space.shape[0],
            hidden_sizes=hidden_sizes,
        )
    elif isinstance(env.action_space, gym.spaces.Discrete):
        actor = CategoricalPolicyNet(
            rngs=rngs,
            observation_dims=env.observation_space.shape[0],
            action_dims=env.action_space.n,
            hidden_sizes=hidden_sizes,
        )
    critic = ValueNet(
        rngs=rngs,
        observation_dims=env.observation_space.shape[0],
        hidden_sizes=hidden_sizes,
    )
    actor_optimizer = nnx.Optimizer(actor, optax.adamw(actor_lr))
    critic_optimizer = nnx.Optimizer(critic, optax.adamw(critic_lr))
    buffer = A2CBuffer([], [], [], [], [], [])
    journal_learn = {
        "episode_idx": 0,
        "step_idx": 0,
        "episode_len": [0],
        "deposit_return": [0.0],
        "averaged_return": [],
    }
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    # LOOP
    last_obs, info = env.reset()
    for e in range(max_epochs):
        for st in range((min_epoch_episodes + 1) * env.spec.max_episode_steps):
            # Play a step
            pred_act, last_val = resolve_and_assess(rngs, actor, critic, last_obs)
            act = (
                pred_act.squeeze()
                if isinstance(env.action_space, gym.spaces.Discrete)
                else pred_act
            )
            next_obs, rew, term, trunc, info = env.step(np.array(act))
            # buffer.store_step(last_obs, act, rew, last_val)
            buffer.store_step(last_obs, pred_act, rew, last_val)
            last_obs = next_obs.copy()
            # Step statistics
            journal_learn["step_idx"] += 1  # TODO: update journal in a util function
            journal_learn["episode_len"][-1] += 1
            journal_learn["deposit_return"][-1] += rew
            # Wrap up episode
            if term or trunc:
                if trunc:
                    eoe_val = jnp.squeeze(critic(last_obs))
                else:
                    eoe_val = jnp.zeros(shape=())
                buffer.wrapup_episode(
                    eoe_val, journal_learn["episode_len"][-1], discount, tradeoff
                )
                # Episode statistics
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
        obj_inv = update_actor_params(actor, actor_optimizer, experience_batch)
        # print(f"Policy objective: {-obj_inv}")  # TODO: Metrics
        for _ in range(critic_update_iters):
            v_loss = update_critic_params(critic, critic_optimizer, experience_batch)
            # print(f"Value estimation loss: {v_loss}")  # TODO: Metrics
        buffer = A2CBuffer([], [], [], [], [], [])
        if not (e + 1) % save_per_epoch or (e + 1) == max_epochs:
            save_models(
                ckpt_dir,
                e + 1,
                actor,
                critic,
                checkpointer,
            )
    # TODO: need a plotter
    plt.plot(journal_learn["averaged_return"])
    plt.grid(visible=True)
    plt.show()
    # plt.savefig(Path(__file__).parent.joinpath(f"{env_name}.png"))

    # Evaluation
    if eval_flag:  # TODO: save and load
        env_options["render_mode"] = "human"
        env = gym.make(env_name, **env_options)
        obs, _ = env.reset()
        episode_return = 0.0
        for _ in range(env.spec.max_episode_steps):
            pred_act, _ = resolve_and_assess(rngs, actor, critic, obs)
            act = (
                pred_act.squeeze()
                if isinstance(env.action_space, gym.spaces.Discrete)
                else pred_act
            )
            obs, rew, term, trunc, _ = env.step(np.array(act))
            episode_return += rew
            if term or trunc:
                print(f"\n---return: {episode_return}---\n")
                break


def play(
    env_name: str = "CartPole-v1",
    env_options: Optional[dict] = {"render_mode": "human"},
    seed: int = 25,
    hidden_sizes: tuple = (64, 64),
    num_episodes: int = 1,  # minimal episodes per epoch
    ckpt_dir: PosixPath = Path("/tmp/spinupax/a2c/checkpoints/"),
    load_epoch_idx: int = 32,
):
    # SETUP
    env = gym.make(env_name, **env_options)
    rngs = nnx.Rngs(seed)
    ## Load models
    if isinstance(env.action_space, gym.spaces.Box):
        dummy_actor = GaussianPolicyNet(
            rngs=rngs,
            observation_dims=env.observation_space.shape[0],
            action_dims=env.action_space.shape[0],
            hidden_sizes=hidden_sizes,
        )
    elif isinstance(env.action_space, gym.spaces.Discrete):
        dummy_actor = CategoricalPolicyNet(
            rngs,
            observation_dims=env.observation_space.shape[0],
            action_dims=env.action_space.n,
            hidden_sizes=hidden_sizes,
        )
    abstract_actor = nnx.eval_shape(lambda: dummy_actor)
    actor_graphdef, abstract_actor_state = nnx.split(abstract_actor)
    dummy_critic = ValueNet(
        rngs=rngs,
        observation_dims=env.observation_space.shape[0],
        hidden_sizes=hidden_sizes,
    )
    abstract_critic = nnx.eval_shape(lambda: dummy_critic)
    critic_graphdef, abstract_critic_state = nnx.split(abstract_critic)
    actor, critic = load_models(
        ckpt_dir,
        load_epoch_idx,
        actor_graphdef,
        critic_graphdef,
        abstract_actor_state,
        abstract_critic_state,
    )

    # LOOP
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        for _ in range(env.spec.max_episode_steps):
            pred_act, _ = resolve_and_assess(rngs, actor, critic, obs)
            act = (
                pred_act.squeeze()
                if isinstance(env.action_space, gym.spaces.Discrete)
                else pred_act
            )
            obs, rew, term, trunc, _ = env.step(np.array(act))
            episode_return += rew
            if term or trunc:
                print(f"\n---return: {episode_return}---\n")
                break


if __name__ == "__main__":
    # TODO: argparse
    # learn(
    # env_name="CartPole-v1",
    # env_options={"render_mode": "rgb_array"},
    # seed=25,
    # discount=0.99,
    # tradeoff=0.97,
    # max_epochs=32,
    # actor_lr=3e-4,
    # critic_lr=1e-4,
    # critic_update_iters=80,
    # hidden_sizes=(64, 64),
    # min_epoch_episodes=5,  # minimal episodes per epoch
    # eval_flag=False,
    # ckpt_dir=Path("/tmp/spinupax/a2c/checkpoints/"),
    # save_per_epoch=10,
    # )
    play(
        # env_name="CartPole-v1",
        # env_options={"render_mode": "rgb_array"},
        # seed=25,
        # hidden_sizes=(64, 64),
        # num_episodes=1,  # minimal episodes per epoch
        # ckpt_dir=Path("/tmp/spinupax/a2c/checkpoints/"),
        # load_epoch_idx=32,
    )
