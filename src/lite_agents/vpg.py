from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from distrax import Greedy, EpsilonGreedy
from scipy.signal import lfilter


Replay = namedtuple('Replay', ['obs', 'act', 'ret'])

class OnPolicyReplayBuffer(object):
    """A simple on-policy replay buffer."""

    def __init__(self, capacity: int, obs_shape: tuple, act_shape: tuple, num_act=None):
        # Variables
        self.id = 0  # buffer instance index
        self.ep_init_id = 0  # episode initial index
        # Properties
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.num_act = num_act
        # Replay storages
        self.buf_obs = np.zeros(shape=[capacity]+list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=(capacity, 1), dtype=int)
        self.buf_rews = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_rets = np.zeros_like(self.buf_rews)

    def store(self, observation, action, reward):
        assert self.id < self.capacity
        self.buf_obs[self.id] = observation
        self.buf_acts[self.id] = action
        self.buf_rews[self.id] = reward
        self.id += 1

    def finish_episode(self, discount=0.9):
        """ End of episode process
        Call this at the end of a trajectory, to compute the rewards-to-go.
        """
        def compute_rtgs(rews):
            return lfilter([1], [1, -discount], rews[::-1], axis=0,)[::-1]

        ep_slice = slice(self.ep_init_id, self.id)
        self.buf_rets[ep_slice] = compute_rtgs(self.buf_rews)
        self.ep_init_id = self.id

    def extract(self):
        """Get replay experience
        """
        replay = Replay(
            self.buf_obs,
            self.buf_acts,
            self.buf_rets,
        )
        # clean up replay buffer for next epoch
        self.__init__(self.capacity, self.obs_shape, self.act_shape, self.num_act)
        return replay


class MLP(nn.Module):
    num_outputs: int
    hidden_sizes: tuple = (64, 64)

    @nn.compact
    def __call__(self, inputs):
        dtype = jnp.float32
        x = inputs.astype(dtype)
        for i, size in enumerate(self.hidden_sizes):
            z = nn.Dense(features=size, name='hidden'+str(i+1), dtype=dtype)(x)
            x = nn.relu(z)
        logits = nn.Dense(features=self.num_outputs, name='logits')(x)
        return logits

