from collections import namedtuple
import gymnasium as gym
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx

ExperienceBatch = namedtuple(
    "ExperienceBatch", "last_observation action reward discount next_observation"
)


class DQNBuffer:
    def __init__(
        self,
        loc: int = 0,
        max_cap: int = int(1e6),
        obs_dims: int = 4,
        discount: float = 0.99,
    ):
        self.lobs_buf = np.zeros((max_cap, obs_dims))
        self.act_buf = np.zeros((max_cap, 1))
        self.rew_buf = np.zeros((max_cap, 1))
        self.adjdisc_buf = np.zeros((max_cap, 1))
        self.nobs_buf = np.zeros_like(self.lobs_buf)
        self.loc = loc
        self.discount = discount

    def store_step(self, last_obs, act, rew, term, next_obs):
        self.lobs_buf[self.loc] = last_obs
        self.act_buf[self.loc] = act
        self.rew_buf[self.loc] = rew
        self.adjdisc_buf[self.loc] = (1 - term) * self.discount
        self.nobs_buf[self.loc] = next_obs

    def extract_experience(self, batch_size, discount=0.98):
        shuffled_ids = np.random.randint(low=0, high=self.capacity, size=(batch_size,))
        observations_batch = jnp.array(self.observations)  # NOTE: won't work under 1D
        actions_batch = jnp.array(self.actions)
        returns_batch = jnp.expand_dims(jnp.array(self.step_returns), axis=-1)
        advantages_batch = jnp.expand_dims(jnp.array(self.advantages), axis=-1)
        log_probas_batch = jnp.array(self.log_probas)

        experience_batch = ExperienceBatch(
            last_observation_samples,
            action_samples,
            reward_samples,
            discount_samples,
            next_observation_samples,
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
qnet = QValueNet(rngs=rngs)
explore_epsilon = 0.5
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
    pred_act, q_val = resolve_and_assess(rngs, qnet, explore_epsilon, last_obs)
    act = pred_act.squeeze()
    next_obs, rew, term, trunc, info = env.step(np.array(act))
    # buffer.store_step(last_obs, pred_act, rew, term, next_obs)
    last_obs = next_obs.copy()
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
