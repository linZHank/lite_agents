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
        self.linear1 = nnx.Linear(2, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 128, rngs=rngs)
        self.linear3 = nnx.Linear(128, 3, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))  # 1st layer
        x = nnx.relu(self.linear2(x))  # 1st layer
        y = self.linear3(x)  # 1st layer
        return y


@nnx.jit
def make_decision(model: PolicyNet, rngs: nnx.Rngs, obs: np.ndarray):
    logits = nnx.log_softmax(model(obs))
    distribution = tfp.distributions.Categorical(logits=logits)
    act = distribution.sample(seed=rngs)
    logp_a = distribution.log_prob(act)
    return act, logp_a


@nnx.jit
def objective_fn(actor: PolicyNet, experience_batch: ExperienceBatch):
    logits = nnx.log_softmax(actor(experience_batch.obs))
    distr = tfp.distributions.Categorical(logits=logits)
    log_pi_a = distr.log_prob(experience_batch.act)
    objective_value = experience_batch.ret * log_pi_a  # expected return

    return -objective_value.mean()


@nnx.jit
def update_params(actor, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(objective_fn)
    objective, grads = grad_fn(actor, experience_batch)
    optimizer.update(grads)  # In-place updates.


# SETUP
env = gym.make("MountainCar-v0", render_mode="rgb_array")
last_obs, _ = env.reset()
prng_keys = nnx.Rngs(29)
buffer = VPGBuffer([], [], [], [])
actor = PolicyNet(rngs=prng_keys)
optimizer = nnx.Optimizer(actor, optax.adamw(3e-4, 0.9))
max_epochs = 256
num_episodes, num_steps = 0, 0
len_episode = 0
episode_return = 0.0
deposit_return, average_return = [], []


# LOOP
for e in range(max_epochs):
    for st in range(2000 + env.spec.max_episode_steps):  # iterate epoch steps
        # act = env.action_space.sample()
        act, logp = make_decision(actor, prng_keys, last_obs)
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
            if st > 2000:  # let epoch end at a finished episode
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
plt.ylim(-200, 0)
plt.yticks(np.arange(-200, 0, 20))
plt.grid(visible=True, axis="y")
plt.savefig(Path(__file__).parent.joinpath("vpg.png"))


# VALIDATION
input("Press any key to evaluate agent")
env = gym.make("MountainCar-v0", render_mode="human")
last_obs, _ = env.reset()
episode_return = 0.0
term, trunc = False, False
for _ in range(env.spec.max_episode_steps):
    act, _ = make_decision(actor, prng_keys, last_obs)
    next_obs, rew, term, trunc, _ = env.step(int(act))
    episode_return += rew
    last_obs = next_obs
    if term or trunc:
        print(f"\n---return: {episode_return}---\n")
        break
env.close()
