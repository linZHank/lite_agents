import gymnasium as gym
from collections import namedtuple
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from tensorflow_probability.substrates.jax.distributions import Normal
import matplotlib.pyplot as plt
from scipy.signal import lfilter


ReplayBuffer = namedtuple(
    "ReplayBuffer",
    "observations actions rewards values step_returns advantages",
)
ExperienceBatch = namedtuple("ExperienceBatch", "obs act ret adv")


class ACBuffer(ReplayBuffer):
    def store_step(self, obs, act, rew, val):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.values.append(val)

    def wrapup_episode(self, eoe_value, episode_len, discount=0.99, tradeoff=0.97):
        """
        Process raw data after an episode, calculate returns-to-go and GAE advantage estimations.
        Args:
            eoe_value: end of episode value estimation.
            episode_len: number of steps in the just-finished episode.
            discount (gamma): price of a reward in future will be penalized at present.
            tradeoff (lambda): balance variance and bias of advantage estimation.
                GAE(lambda=0): r_t + gamma V(s_{t+1}) - V(s_t), high bias low variance
                GAE(lambda=1): sum_{l=0}^{infty} gamma^l r+{t+1} - V(s_t), low bias high variance
        """
        next_vals = self.values[-episode_len + 1 :]
        next_vals.append(eoe_value)
        nv_arr = jnp.array(next_vals)
        r_arr = jnp.array(self.rewards[-episode_len:])
        v_arr = jnp.array(self.values[-episode_len:])
        # GAE-Lambda advantage
        td_errs = r_arr + discount * nv_arr - v_arr  # r_t + gamma V(s_{t+1}) - V(s_t)
        gae_advs = jnp.flip(
            lfilter([1], [1, -discount * tradeoff], jnp.flip(td_errs)), axis=0
        )
        self.advantages.extend(gae_advs.tolist())
        # Discounted returns to-go
        rev_ep_rews = self.rewards[-episode_len:][::-1]  # reversed episodic rewards
        drtg = lfilter([1], [1, -discount], rev_ep_rews)[::-1]  # discounted return togo
        self.step_returns.extend(drtg.tolist())

    def extract_experience(self):
        observations_batch = jnp.array(self.observations)  # NOTE: won't work under 1D
        actions_batch = jnp.array(self.actions)
        returns_batch = jnp.expand_dims(jnp.array(self.step_returns), axis=-1)
        advantages_batch = jnp.expand_dims(jnp.array(self.advantages), axis=-1)

        experience_batch = ExperienceBatch(
            observations_batch, actions_batch, returns_batch, advantages_batch
        )

        return experience_batch


class PolicyNet(nnx.Module):
    """MLP Categorical actor"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(8, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        mu = self.linear3(x)
        log_sigma = self.linear3(x)  # log(pi(a|s))
        pi = Normal(loc=mu, scale=jnp.exp(log_sigma))

        return pi


class ValueNet(nnx.Module):
    """MLP critic"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(8, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        v = self.linear3(x)

        return v


@nnx.jit
def objective_fn(actor: PolicyNet, experience_batch: ExperienceBatch):
    policies = actor(experience_batch.obs)
    log_pi_as = policies.log_prob(experience_batch.act)
    objectives = experience_batch.adv * log_pi_as  # expected return

    return -objectives.mean()


@nnx.jit
def loss_fn(critic: ValueNet, experience_batch: ExperienceBatch):
    pred_vals = critic(experience_batch.obs)
    targ_vals = experience_batch.ret
    mse_loss = (pred_vals - targ_vals) ** 2  # TODO: use MSE from a lib

    return mse_loss.mean()


@nnx.jit
def update_actor_params(actor, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(objective_fn)
    inv_objectives, grads = grad_fn(actor, experience_batch)
    optimizer.update(grads)  # In-place updates.

    return inv_objectives


@nnx.jit
def update_critic_params(critic, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(loss_fn)
    v_loss, grads = grad_fn(critic, experience_batch)
    optimizer.update(grads)  # In-place updates.

    return v_loss


@nnx.jit
def resolve_and_assess(
    rngs: nnx.Rngs, actor: PolicyNet, critic: ValueNet, obs: np.ndarray
):
    pi = actor(jnp.expand_dims(obs, axis=0))
    act = pi.sample(seed=rngs)
    val = critic(obs)

    return act.squeeze(axis=0), val.squeeze()


# SETUP
env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
rngs = nnx.Rngs(25)
actor = PolicyNet(rngs=rngs)
critic = ValueNet(rngs=rngs)
nnx.display(actor)
nnx.display(critic)
optimizer_actor = nnx.Optimizer(actor, optax.adamw(3e-4))
optimizer_critic = nnx.Optimizer(actor, optax.adamw(1e-4))
buffer = ACBuffer([], [], [], [], [], [])
journal = {
    "episode_idx": 0,
    "step_idx": 0,
    "episode_len": [0],
    "deposit_return": [0.0],
    "averaged_return": [],
}

last_obs, info = env.reset()
for e in range(2):  # try 64
    for st in range(6 * env.spec.max_episode_steps):
        # Play a step
        act, last_val = resolve_and_assess(rngs, actor, critic, last_obs)
        next_obs, rew, term, trunc, info = env.step(np.array(act.squeeze()))
        buffer.store_step(last_obs, act, rew, last_val)
        last_obs = next_obs.copy()
        # Step statistics
        journal["step_idx"] += 1  # TODO: update journal in a util function
        journal["episode_len"][-1] += 1
        journal["deposit_return"][-1] += rew
        # Wrap up espisode
        if term or trunc:
            if trunc:
                eoe_val = jnp.squeeze(critic(last_obs))
            else:
                eoe_val = jnp.zeros(shape=())
            buffer.wrapup_episode(eoe_val, journal["episode_len"][-1])
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
                break  # end epoch after sufficient data collected
    # Wrap up epoch
    exp_batch = buffer.extract_experience()
    obj_inv = update_actor_params(actor, optimizer_actor, exp_batch)
    print(f"Policy objective: {-obj_inv}")
    for _ in range(50):
        v_loss = update_critic_params(critic, optimizer_critic, exp_batch)
        print(f"Value estimation loss: {v_loss}")
    buffer = ACBuffer([], [], [], [], [], [])
    print(
        f"===\nepoch {e + 1} \n\ttotal steps: {journal['step_idx']}\n\taveraged return: {journal['averaged_return'][-1]}\n==="
    )

plt.plot(journal["averaged_return"])
# plt.ylim(0, 200)
# plt.yticks(np.arange(0, 200, 20))
plt.grid(visible=True)
plt.savefig(Path(__file__).parent.joinpath("vac_continuous.png"))


# VALIDATION
# input("Press any key to evaluate agent")
# env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
# obs, _ = env.reset()
# episode_return = 0.0
# term, trunc = False, False
# for _ in range(env.spec.max_episode_steps):
#     act, _ = resolve_and_assess(rngs, actor, critic, obs)
#     obs, rew, term, trunc, _ = env.step(np.array(act.squeeze()))
#     episode_return += rew
#     if term or trunc:
#         print(f"\n---Evaluation episode return: {episode_return}---\n")
#         break
