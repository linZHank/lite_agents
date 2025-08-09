from collections import namedtuple
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx


ExperienceBatch = namedtuple("ExperienceBatch", "lobs act rew disct nobs")


class DQNBuffer:
    def __init__(
        self,
        observation_dims: int,
        discount: float = 0.99,
        capacity: int = int(1e5),
    ):
        self.lobs_buf = np.zeros((capacity, observation_dims))
        self.act_buf = np.zeros((capacity, 1))
        self.rew_buf = np.zeros((capacity, 1))
        self.disct_buf = np.zeros((capacity, 1))
        self.nobs_buf = np.zeros_like(self.lobs_buf)
        # Vars
        self.occupancy = 0
        self.loc = 0
        # Constants
        self.capacity = capacity
        self.discount = discount

    def store_step(self, last_obs, act, rew, term, next_obs):
        self.lobs_buf[self.loc] = last_obs
        self.act_buf[self.loc] = act
        self.rew_buf[self.loc] = rew
        self.disct_buf[self.loc] = (1 - term) * self.discount
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


class QValueNet(nnx.Module):
    """Q-Value Network"""

    def __init__(
        self,
        rngs: nnx.Rngs,
        observation_dims: int,
        action_dims: int,
        hidden_sizes: tuple = (64, 64),
    ):
        self.backbone_sizes = (observation_dims, *hidden_sizes)
        self.backbone_transforms = []  # TODO: functionize
        for i in range(len(self.backbone_sizes) - 1):
            self.backbone_transforms.append(
                nnx.Linear(
                    self.backbone_sizes[i], self.backbone_sizes[i + 1], rngs=rngs
                )
            )
        self.output_transform = nnx.Linear(
            self.backbone_sizes[-1], action_dims, rngs=rngs
        )

    def __call__(self, x):
        for trans in self.backbone_transforms:
            x = nnx.relu(trans(x))
        q_value = self.output_transform(x)

        return q_value
