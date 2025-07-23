import gymnasium as gym
from collections import namedtuple
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from tensorflow_probability.substrates.jax import distributions
from tensorflow_probability.substrates.jax.distributions import Categorical
import matplotlib.pyplot as plt
from scipy.signal import lfilter


ReplayBuffer = namedtuple(
    "ReplayBuffer",
    "observations actions rewards values step_returns advantages",
)
ExperienceBatch = namedtuple("ExperienceBatch", "obs act ret adv")


class VACBuffer(ReplayBuffer):
    def store_step(self, obs, act, rew, val):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.values.append(val)

    def wrapup_episode(self, val_eoe, len_episode, discount=0.99, lam=0.95):
        next_vals = self.values[-len_episode + 1 :]
        next_vals.append(val_eoe)
        nv_arr = jnp.array(next_vals)
        r_arr = jnp.array(self.rewards[-len_episode:])
        v_arr = jnp.array(self.values[-len_episode:])
        # GAE-Lambda advantage
        td_errs = r_arr + discount * nv_arr - v_arr  # r_t + gamma * V_{t+1} - V_t
        gae_advs = jnp.flip(lfilter([1], [1, -gamma * lam], jnp.flip(td_errs)), axis=0)
        self.advantages.extend(gae_advs.tolist())
        # Discounted returns to-go
        rev_ep_rews = self.rewards[-len_episode:][::-1]  # reversed episodic rewards
        drtg = lfilter([1], [1, -discount], rev_ep_rews)[::-1]  # discounted return togo
        self.step_returns.extend(drtg.tolist())

    def extract_experience(self):
        observations_batch = jnp.array(self.observations)
        actions_batch = jnp.array(self.actions)
        returns_batch = jnp.array(self.step_returns)
        advantages_batch = jnp.array(self.advantages)

        experience_batch = ExperienceBatch(
            observations_batch, actions_batch, returns_batch, advantages_batch
        )

        return experience_batch


class PolicyNet(nnx.Module):
    """MLP Categorical actor"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 128, rngs=rngs)
        self.linear3 = nnx.Linear(128, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))  # 1st layer
        x = nnx.relu(self.linear2(x))  # 1st layer
        y = self.linear3(x)  # 1st layer
        log_prob = nnx.log_softmax(y)  # log(pi(a|s))
        pi = Categorical(logits=log_prob)

        return pi


class ValueNet(nnx.Module):
    """MLP critic"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 128, rngs=rngs)
        self.linear3 = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))  # 1st layer
        x = nnx.relu(self.linear2(x))  # 1st layer
        v = self.linear3(x)  # 1st layer
        return v


@nnx.jit
def make_decision(rngs: nnx.Rngs, actor: PolicyNet, critic: ValueNet, obs: np.ndarray):
    policy = actor(obs)
    act = policy.sample(seed=rngs)
    val = critic(obs)
    return act, val


@nnx.jit
def objective_fn(actor: PolicyNet, experience_batch: ExperienceBatch):
    policies = actor(experience_batch.obs)
    log_pi_as = policies.log_prob(experience_batch.act)
    objectives = experience_batch.adv * log_pi_as  # expected return

    return -objectives.mean()


@nnx.jit
def loss_fn(ciritic: ValueNet, experience_batch: ExperienceBatch):
    vals = critic(experience_batch.obs)
    v_loss = (vals - experience_batch.ret) ** 2

    return v_loss.mean()


@nnx.jit
def update_actor_params(actor, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(objective_fn)
    objective, grads = grad_fn(actor, experience_batch)
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def update_critic_params(critic, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(loss_fn)
    v_loss, grads = grad_fn(actor, experience_batch)
    optimizer.update(grads)  # In-place updates.


# SETUP
env = gym.make("CartPole-v1", render_mode="rgb_array")
rngs = nnx.Rngs(25)
buffer = VACBuffer([], [], [], [], [], [])
actor = PolicyNet(rngs=rngs)
critic = ValueNet(rngs=rngs)
nnx.display(actor)
nnx.display(critic)
optimizer_actor = nnx.Optimizer(actor, optax.adamw(3e-4))
optimizer_critic = nnx.Optimizer(actor, optax.adamw(1e-4))
journal = {
    "episode_idx": 0,
    "step_idx": 0,
    "episode_len": [0],
    "deposit_return": [0.0],
    "averaged_return": [],
}
last_obs, info = env.reset()

for e in range(4):
    for st in range(6 * env.spec.max_episode_steps):
        act, last_val = make_decision(rngs, actor, critic, last_obs)
        next_obs, rew, term, trunc, info = env.step(np.array(act))
        buffer.store_step(last_obs, act, rew, last_val)
        # TODO: update journal in a util function
        journal["step_idx"] += 1
        journal["episode_len"][-1] += 1
        journal["deposit_return"][-1] += rew
        last_obs = next_obs.copy()
        if term or trunc:
            # buffer.wrapup_episode(end_val, journal["episode_len"][-1])
            # Episode statistics
            journal["episode_idx"] += 1
            journal["averaged_return"].append(
                sum(journal["deposit_return"]) / len(journal["deposit_return"])
            )
            # TODO: need a logger
            print(
                f"---\nepisode: {journal['episode_idx']}, length: {journal['episode_len'][-1]}, return: {journal['deposit_return'][-1]}\n---\n"
            )
            # Reset episode
            last_obs, _ = env.reset()
            journal["episode_len"].append(0)
            journal["deposit_return"].append(0.0)
            if st > 5 * env.spec.max_episode_steps:
                break
    # Epoch statistics
    print(
        f"===\nepoch {e + 1} \n\ttotal steps: {journal['step_idx']}\n\taveraged return: {journal['averaged_return'][-1]}\n==="
    )
    exp_batch = buffer.extract_experience()
    # update_params(actor, optimizer, exp_batch)
    buffer = VACBuffer([], [], [], [], [], [])

plt.plot(journal["averaged_return"])
plt.grid(visible=True, axis="y")
plt.show()
# plt.savefig(Path(__file__).parent.joinpath("vpg.png"))


# VALIDATION
input("Press any key to evaluate agent")
env = gym.make("CartPole-v1", render_mode="human")
last_obs, _ = env.reset()
episode_return = 0.0
term, trunc = False, False
for _ in range(env.spec.max_episode_steps):
    act, val = make_decision(rngs, actor, critic, last_obs)
    next_obs, rew, term, trunc, _ = env.step(int(act))
    episode_return += rew
    last_obs = next_obs
    if term or trunc:
        print(f"\n---return: {episode_return}---\n")
        break
env.close()
