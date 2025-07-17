import gymnasium as gym
from collections import namedtuple
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
from scipy.signal import lfilter


ReplayBuffer = namedtuple("ReplayBuffer", "observations actions rewards step_returns")
ExperienceBatch = namedtuple("ExperienceBatch", "obs act ret")


class VPGBuffer(ReplayBuffer):
    def store_step(self, obs, act, rew):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)

    def wrapup_episode(self, len_episode, discount=0.98):
        # self.step_returns.extend([sum(self.rewards[-len_episode:])] * len_episode)
        rev_ep_rews = self.rewards[-len_episode:][::-1]  # reversed episodic rewards
        drtg = lfilter([1], [1, -discount], rev_ep_rews)[::-1]  # discounted return togo
        self.step_returns.extend(drtg.tolist())

    def extract_experience(self):
        observations_batch = jnp.array(self.observations)
        actions_batch = jnp.array(self.actions)
        returns_batch = jnp.array(self.step_returns)

        experience_batch = ExperienceBatch(
            observations_batch, actions_batch, returns_batch
        )

        return experience_batch


class PolicyNet(nnx.Module):
    """A simple fully-connected Neural Network model"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(3, 128, rngs=rngs)  # 3-dim o_space
        self.linear2 = nnx.Linear(128, 128, rngs=rngs)
        self.linear3 = nnx.Linear(128, 1, rngs=rngs)  # 1-dim a_space

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))  # 1st layer
        x = nnx.relu(self.linear2(x))  # 2nd layer
        mu = self.linear3(x)  # mean
        log_sigma = self.linear3(x)
        return mu, log_sigma


@nnx.jit
def make_decision(rngs: nnx.Rngs, model: PolicyNet, obs: np.ndarray):
    mu, log_sigma = actor(obs)
    distribution = tfp.distributions.Normal(loc=mu, scale=jnp.exp(log_sigma))
    sampled_action = distribution.sample(seed=rngs)
    # logpi_agvns = distribution.log_prob(sampled_action)
    return sampled_action, mu, log_sigma


@nnx.jit
def objective_fn(actor: PolicyNet, experience_batch: ExperienceBatch):
    mu_batch, log_sigma_batch = actor(experience_batch.obs)
    distr_batch = tfp.distributions.Normal(loc=mu_batch, scale=jnp.exp(log_sigma_batch))
    log_pi_as = distr_batch.log_prob(experience_batch.act)
    objective_value = experience_batch.ret * log_pi_as  # expected return

    return -objective_value.mean()


@nnx.jit
def update_params(actor, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(objective_fn)
    obj_val, grads = grad_fn(actor, experience_batch)
    optimizer.update(grads)  # In-place updates.


# SETUP
env = gym.make("Pendulum-v1", render_mode="rgb_array")
last_obs, _ = env.reset()
prng_keys = nnx.Rngs(19)
buffer = VPGBuffer([], [], [], [])
actor = PolicyNet(rngs=prng_keys)
optimizer = nnx.Optimizer(actor, optax.adamw(1e-4, 0.95))
max_epochs = 512
num_episodes, num_steps = 0, 0
len_episode = 0
episode_return = 0.0
deposit_return, average_return = [], []


# LOOP
for e in range(max_epochs):
    for st in range(11 * env.spec.max_episode_steps):  # at least 10 finished episodes
        # act = env.action_space.sample()
        act, mean, log_stdd = make_decision(prng_keys, actor, last_obs)
        # print(act, logp)
        next_obs, rew, term, trunc, info = env.step(np.array(act))
        # Step statistics
        # print("\n")
        # print(f"last observation: {last_obs}")
        # print(f"action: {act}")
        # print(f"next observation: {next_obs}")
        # print(f"reward: {rew}")
        # print(f"episode terminated: {term}")
        # print(f"episode truncated: {trunc}")
        # print(f"info: {info}")
        # print("\n")
        buffer.store_step(last_obs, act, rew)
        episode_return += rew
        num_steps += 1
        len_episode += 1
        last_obs = next_obs
        if term or trunc:
            buffer.wrapup_episode(len_episode)
            # Episode statistics
            num_episodes += 1
            deposit_return.append(episode_return)
            average_return.append(sum(deposit_return) / len(deposit_return))
            print(
                f"\n---\nepisode: {num_episodes}, length: {len_episode}, return: {episode_return}\n---\n"
            )
            # Reset episode
            len_episode, episode_return = 0, 0
            last_obs, _ = env.reset()
            if st > 10 * env.spec.max_episode_steps:  # finish last episode
                break
    # Epoch statistics
    print(
        f"\n===\nepoch {e + 1} \n\ttotal steps: {num_steps}\n\taveraged return: {average_return[-1]}\n==="
    )
    # Update actor
    experience_batch = buffer.extract_experience()
    # exp_ret = objective_fn(actor, experience_batch)
    update_params(actor, optimizer, experience_batch)
    buffer = VPGBuffer([], [], [], [])


plt.plot(average_return)
# plt.ylim(-500, -100)
# plt.yticks(np.arange(-500, -100, 50))
plt.grid(visible=True)
plt.savefig(Path(__file__).parent.joinpath("vpg.png"))


# VALIDATION
input("Press any key to evaluate agent")
env = gym.make("Pendulum-v1", render_mode="human")
last_obs, _ = env.reset()
episode_return = 0.0
term, trunc = False, False
for _ in range(env.spec.max_episode_steps):
    act_sample, mean, log_stdd = make_decision(prng_keys, actor, last_obs)
    # next_obs, rew, term, trunc, _ = env.step(np.array(mean))
    next_obs, rew, term, trunc, _ = env.step(np.array(act_sample))
    episode_return += rew
    last_obs = next_obs
    if term or trunc:
        print(f"\n---return: {episode_return}---\n")
        break
