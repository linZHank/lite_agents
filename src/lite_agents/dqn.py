from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


Batch = namedtuple('Batch', ['pobs', 'acts', 'discrews', 'nobs'])


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity, dim_obs):
        self.pobs_buf = np.zeros(shape=[capacity]+list(dim_obs), dtype=np.float32)
        self.acts_buf = np.zeros(shape=capacity, dtype=int)
        self.rews_buf = np.zeros(shape=capacity, dtype=np.float32)
        self.nobs_buf = np.zeros_like(self.pobs_buf)
        self.termsigs_buf = np.zeros(shape=capacity, dtype=np.float32)
        # variables
        self.loc = 0  # replay instance index
        self.buffer_size = 0
        # property
        self.capacity = capacity

    def store(self, prev_obs, action, reward, term_signal, next_obs):
        self.pobs_buf[self.loc] = prev_obs
        self.acts_buf[self.loc] = action
        self.rews_buf[self.loc] = reward
        self.nobs_buf[self.loc] = next_obs
        self.termsigs_buf[self.loc] = term_signal
        self.loc = (self.loc + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self, batch_size, discount_factor=0.99):
        indices = np.random.randint(low=0, high=self.buffer_size, size=(batch_size,))
        pobs_samples = jnp.asarray(self.pobs_buf[indices])
        nobs_samples = jnp.asarray(self.nobs_buf[indices])
        acts_samples = jnp.asarray(self.acts_buf[indices])
        rews_samples = jnp.asarray(self.rews_buf[indices])
        termsigs_samples = jnp.asarray(self.termsigs_buf[indices])
        drews_samples = rews_samples * (1 - termsigs_samples) * discount_factor
        sampled_batch = Batch(
            pobs_samples,
            acts_samples,
            # rews_samples,
            # termsigs_samples,
            drews_samples,  # discounted rewards
            nobs_samples
        )
        return sampled_batch

    def is_ready(self, batch_size):  # warm up trick
        return batch_size <= self.capacity


class MLP(nn.Module):
    """Multi-Layer Perceptron model

    """

    num_outputs: int
    hidden_sizes: list

    @nn.compact
    def __call__(self, inputs):
        """Forward pass

        """
        dtype = jnp.float32
        X = inputs.astype(dtype)
        for i, size in enumerate(self.hidden_sizes):
            Z = nn.Dense(features=size, name='hidden'+str(i+1), dtype=dtype)(X)
            X = nn.relu(Z)
        logits = nn.Dense(features=self.num_outputs, name='logits')(X)
        return logits


class DQNAgent:
    """RL agent powered by Deep-Q Network

    """
    def __init__(self, obs_shape, num_actions, hidden_sizes,) -> None:
        self.qnet_online = MLP(num_actions, hidden_sizes)


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        (5, 3),
    )
    print(
        agent.qnet_online.tabulate(
            jax.random.key(0),
            env.observation_space.sample(),
            compute_flops=True,
            compute_vjp_flops=True
        )
    )

