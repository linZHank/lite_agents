from collections import namedtuple
import gymnasium as gym
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import optax

ExperienceBatch = namedtuple("ExperienceBatch", "lobs act rew disct nobs")


class DQNBuffer:
    def __init__(
        self,
        obs_dims: int = 4,
        discount_rate: float = 0.99,
        capacity: int = int(1e4),
    ):
        self.lobs_buf = np.zeros((capacity, obs_dims))
        self.act_buf = np.zeros((capacity, 1))
        self.rew_buf = np.zeros((capacity, 1))
        self.disct_buf = np.zeros((capacity, 1))
        self.nobs_buf = np.zeros_like(self.lobs_buf)
        # Vars
        self.occupancy = 0
        # Constants
        self.capacity = capacity
        self.discount_rate = discount_rate

    def store_step(self, last_obs, act, rew, term, next_obs):
        self.lobs_buf[self.occupancy] = last_obs
        self.act_buf[self.occupancy] = act
        self.rew_buf[self.occupancy] = rew
        self.disct_buf[self.occupancy] = (1 - term) * self.discount_rate
        self.nobs_buf[self.occupancy] = next_obs
        self.occupancy = (self.occupancy + 1) % self.capacity

    def extract_experience(self, rngs, batch_size):
        shuffled_inds = jax.random.choice(
            key=rngs.params(), a=jnp.arange(self.occupancy), shape=(batch_size,)
        )
        lobs_samples = self.lobs_buf[shuffled_inds]
        act_samples = self.act_buf[shuffled_inds]
        rew_samples = self.rew_buf[shuffled_inds]
        disct_samples = self.disct_buf[shuffled_inds]
        nobs_samples = self.nobs_buf[shuffled_inds]

        experience_batch = ExperienceBatch(
            jnp.array(lobs_samples),
            jnp.array(act_samples),
            jnp.array(rew_samples),
            jnp.array(disct_samples),
            jnp.array(nobs_samples),
        )

        return experience_batch


# SETUP
class QValueNet(nnx.Module):
    """MLP Critic"""

    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        q = self.linear3(x)

        return q


@jax.vmap
def double_q_error(data, q_pred, q_next, q_duel):
    q_target = jax.lax.stop_gradient(
        data.rew + data.disct * q_next[q_duel.argmax(axis=-1)]
    )
    td_error = q_target - q_pred[data.act]
    return td_error


@nnx.jit
def loss_fn(critic_online, critic_stable, experience_batch):
    qval_pred = critic_online(experience_batch.lobs)
    qval_next = critic_stable(experience_batch.nobs)
    qval_duel = critic_online(experience_batch.nobs)
    qerr = double_q_error(
        experience_batch,
        qval_pred,
        qval_next,
        qval_duel,
    )
    loss_value = optax.l2_loss(qerr).mean()
    return loss_value


@nnx.jit
def resolve_and_assess(rngs, critic, explore_epsilon, obs):
    # pred_act, q_val = resolve_and_assess(rngs, explore_epsilon, critic, last_obs)
    q_value = critic(obs)
    greedy_action = q_value.argmax(axis=-1)
    sampled_action = jax.random.randint(key=rngs.params(), shape=(), minval=0, maxval=2)
    selector = jax.random.uniform(key=rngs.params())
    action = jax.lax.select(
        pred=selector > explore_epsilon,
        on_true=greedy_action,
        on_false=sampled_action,
    )

    return action, q_value


rngs = nnx.Rngs(25)
qnet_online = QValueNet(rngs=rngs)
qnet_stable = QValueNet(rngs=rngs)
epsilon = 0.999
warmup_episodes = 5
buffer = DQNBuffer()
journal_learn = {
    "episode_idx": 0,
    "step_idx": 0,
    "episode_len": [0],
    "deposit_return": [0.0],
    "averaged_return": [],
}


# LOOP
env = gym.make("CartPole-v1", render_mode="rgb_array")
last_obs, info = env.reset()
for st in range(5 * env.spec.max_episode_steps):
    # Play a step
    pred_act, q_val = resolve_and_assess(rngs, qnet_online, epsilon, last_obs)
    act = pred_act.squeeze()
    next_obs, rew, term, trunc, info = env.step(np.array(act))
    buffer.store_step(last_obs, act, rew, term, next_obs)
    last_obs = next_obs.copy()
    if journal_learn["episode_idx"] + 1 > warmup_episodes:
        experience_batch = buffer.extract_experience(rngs, 1024)
        qloss = loss_fn(qnet_online, qnet_stable, experience_batch)
    # Step statistics
    journal_learn["step_idx"] += 1  # TODO: update journal in a util function
    journal_learn["episode_len"][-1] += 1
    journal_learn["deposit_return"][-1] += rew
    if term or trunc:
        journal_learn["episode_idx"] += 1
        journal_learn["averaged_return"].append(
            sum(journal_learn["deposit_return"]) / len(journal_learn["deposit_return"])
        )
        # TODO: need a logger
        print(
            f"---\nepisode: {journal_learn['episode_idx']}, length: {journal_learn['episode_len'][-1]}, return: {journal_learn['deposit_return'][-1]}\n---\n"
        )
        ep_return = 0
        pobs, _ = env.reset()
        journal_learn["episode_len"].append(0)
        journal_learn["deposit_return"].append(0.0)
