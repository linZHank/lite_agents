from collections import namedtuple
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
from tensorflow_probability.substrates.jax.distributions import Categorical, Normal

from scipy.signal import lfilter


ExperienceBatch = namedtuple("ExperienceBatch", "lobs act rew disct nobs")


class DQNBuffer:
    def __init__(
        self,
        obs_dims: int,
        discount_rate: float = 0.99,
        capacity: int = int(1e5),
    ):
        self.lobs_buf = np.zeros((capacity, obs_dims))
        self.act_buf = np.zeros((capacity, 1))
        self.rew_buf = np.zeros((capacity, 1))
        self.disct_buf = np.zeros((capacity, 1))
        self.nobs_buf = np.zeros_like(self.lobs_buf)
        # Vars
        self.occupancy = 0
        self.loc = 0
        # Constants
        self.capacity = capacity
        self.discount_rate = discount_rate

    def store_step(self, last_obs, act, rew, term, next_obs):
        self.lobs_buf[self.loc] = last_obs
        self.act_buf[self.loc] = act
        self.rew_buf[self.loc] = rew
        self.disct_buf[self.loc] = (1 - term) * self.discount_rate
        self.nobs_buf[self.loc] = next_obs
        self.loc = (self.loc + 1) % self.capacity
        self.occupancy += 1

    def extract_experience(self, rngs, batch_size):
        shuffled_inds = jax.random.choice(
            key=rngs.params(),
            a=jnp.arange(min(self.occupancy, self.capacity)),
            shape=(batch_size,),
        )
        lobs_samples = self.lobs_buf[shuffled_inds]
        act_samples = self.act_buf[shuffled_inds]
        rew_samples = self.rew_buf[shuffled_inds]
        disct_samples = self.disct_buf[shuffled_inds]
        nobs_samples = self.nobs_buf[shuffled_inds]

        experience_batch = ExperienceBatch(
            jnp.array(lobs_samples),
            jnp.array(act_samples, dtype=jnp.int16),
            jnp.array(rew_samples),
            jnp.array(disct_samples),
            jnp.array(nobs_samples),
        )

        return experience_batch


# class MLPNet(nnx.Module):
#     """A simple multi-layer perceptron network"""
#
#     def __init__(
#         self,
#         rngs: nnx.Rngs,
#         input_dims: int,
#         output_dims: int,
#         hidden_sizes: tuple = (64, 64),
#     ):
#         self.backbone_sizes = (input_dims, *hidden_sizes)
#         self.backbone_transforms = []  # TODO: functionize
#         for i in range(len(self.backbone_sizes) - 1):
#             self.backbone_transforms.append(
#                 nnx.Linear(
#                     self.backbone_sizes[i], self.backbone_sizes[i + 1], rngs=rngs
#                 )
#             )
#         self.output_transform = nnx.Linear(
#             self.backbone_sizes[-1], output_dims, rngs=rngs
#         )


# SETUP
class MLPNet(nnx.Module):
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


class CategoricalPolicyNet(MLPNet):
    """Actor for discrete actions space"""

    def __init__(
        self,
        rngs: nnx.Rngs,
        observation_dims: int,
        action_dims: int,
        hidden_sizes: tuple = (64, 64),
    ):
        super().__init__(
            rngs=rngs,
            input_dims=observation_dims,
            output_dims=action_dims,
            hidden_sizes=hidden_sizes,
        )

    def __call__(self, x):
        for trans in self.backbone_transforms:
            x = nnx.relu(trans(x))
        y = self.output_transform(x)
        log_prob = nnx.log_softmax(y)  # log(pi(a|s))
        pi = Categorical(logits=jnp.expand_dims(log_prob, axis=1))

        return pi


class GaussianPolicyNet(MLPNet):
    """Actor for discrete actions space"""

    def __init__(
        self,
        rngs: nnx.Rngs,
        observation_dims: int,
        action_dims: int,
        hidden_sizes: tuple = (64, 64),
    ):
        super().__init__(
            rngs=rngs,
            input_dims=observation_dims,
            output_dims=action_dims,
            hidden_sizes=hidden_sizes,
        )

    def __call__(self, x):
        for trans in self.backbone_transforms:
            x = nnx.relu(trans(x))
        mu = self.output_transform(x)
        log_sigma = self.output_transform(x)
        pi = Normal(loc=mu, scale=jnp.exp(log_sigma))

        return pi


class ValueNet(MLPNet):
    """Critic Net"""

    def __init__(
        self,
        rngs: nnx.Rngs,
        observation_dims: int,
        hidden_sizes: tuple = (64, 64),
    ):
        super().__init__(
            rngs=rngs,
            input_dims=observation_dims,
            output_dims=1,
            hidden_sizes=hidden_sizes,
        )

    def __call__(self, x):
        for trans in self.backbone_transforms:
            x = nnx.relu(trans(x))
        v = self.output_transform(x)
        return v
