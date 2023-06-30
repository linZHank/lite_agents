"""Class and functions to implement a simple DQN agent"""

from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from distrax import Greedy, EpsilonGreedy


Batch = namedtuple('Batch', ['pobs', 'acts', 'rews', 'termsigs', 'nobs'])
Params = namedtuple('Params', 'online, target')


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity, dim_obs):
        self.pobs_stash = np.zeros(shape=[capacity]+list(dim_obs), dtype=np.float32)
        self.acts_stash = np.zeros(shape=capacity, dtype=np.float32)
        self.rews_stash = np.zeros(shape=capacity, dtype=np.float32)
        self.nobs_stash = np.zeros_like(self.pobs_stash)
        self.termsigs_stash = np.zeros(shape=capacity, dtype=np.float32)
        self.loc = 0  # replay instance index
        self.stash_size = 0
        self.capacity = capacity

    def store(self, prev_obs, action, reward, term_signal, next_obs):
        self.pobs_stash[self.loc] = prev_obs
        self.acts_stash[self.loc] = action
        self.rews_stash[self.loc] = reward
        self.nobs_stash[self.loc] = next_obs
        self.termsigs_stash[self.loc] = term_signal
        self.loc = (self.loc + 1) % self.capacity
        self.stash_size = min(self.stash_size + 1, self.capacity)

    def sample(self, batch_size, discount_factor=0.99):
        indices = np.random.randint(low=0, high=self.stash_size, size=(batch_size,))
        pobs_samples = self.pobs_stash[indices]
        nobs_samples = self.nobs_stash[indices]
        acts_samples = self.acts_stash[indices]
        rews_samples = self.rews_stash[indices]
        termsigs_samples = self.termsigs_stash[indices]
        batched_samples = Batch(pobs_samples, acts_samples, rews_samples, termsigs_samples, nobs_samples)
        return batched_samples

    def is_ready(self, batch_size):  # warm up trick
        return batch_size <= self.capacity


class QNet(nn.Module):
    "Q-Net template"

    num_outputs: int

    @nn.compact
    def __call__(self, inputs):
        """Define the basic MLP network architecture

        Network is used to estimate values of state-action pairs
        """
        x = inputs.astype(jnp.float32)
        x = nn.Dense(features=128, name='dense1')(x)
        x = nn.relu(x)
        x = nn.Dense(features=128, name='dense2')(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.num_outputs, name='logits')(x)
        return logits


class DQNAgent:
    """DQN agent template"""

    def __init__(self, observation_shape, num_actions) -> None:
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.qnet = QNet(num_outputs=num_actions)
        self.epsilon_by_frame = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.01,
            power=1,
            transition_steps=500,
        )

    def init_params(self, key, sample_obs):
        online_params = self.qnet.init(key, sample_obs)
        params = Params(online_params, online_params)

        return params

    def make_decision(self, key, params, obs, episode_count, eval_flag=False):
        """pi(a|s)
        TODO:
            add warm up
            rewrite epsilon greedy w/o rlax
        """
        key, subkey = jax.random.split(key)  # generate a new key, or sampled action won't change
        qvals = jnp.squeeze(self.qnet.apply(params, obs))
        epsilon = self.epsilon_by_frame(episode_count)
        sampled_action = EpsilonGreedy(preferences=qvals, epsilon=epsilon).sample(seed=subkey)
        greedy_action = Greedy(preferences=qvals).sample(seed=subkey)
        action = jax.lax.select(eval_flag, greedy_action, sampled_action)

        return key, action, qvals, epsilon


if __name__=='__main__':
    import gymnasium as gym
    env = gym.make('LunarLander-v2')  # , render_mode='human')
    agent = DQNAgent(
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n
    )
    buf = ReplayBuffer(capacity=int(10), dim_obs=env.observation_space.shape)
    key = jax.random.PRNGKey(20)
    qnet_params = agent.init_params(
        key=key,
        sample_obs=env.observation_space.sample()
    )
    # init env
    o_tm1, info = env.reset()
    term, trunc = False, False
    episode_count = 0
    for step in range(200):
        key, a_tm1, qvals, epsilon = agent.make_decision(
            key=key,
            params=qnet_params.online,
            obs=o_tm1,
            episode_count=episode_count,
        )
        o_t, r_t, term, trunc, info = env.step(int(a_tm1))
        buf.store(prev_obs=o_tm1, action=a_tm1, reward=r_t, term_signal=term, next_obs=o_t)
        print(f"\n---episode: {episode_count}, step: {step}, epsilon: {epsilon}---")
        print(f"state: {o_tm1}\naction: {a_tm1}\nreward: {r_t}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {o_t}")
        o_tm1 = o_t
        if term or trunc:
            episode_count += 1
            o_tm1, info = env.reset()
            term, trunc = False, False
            print("reset")
