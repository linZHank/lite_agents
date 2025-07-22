from collections import namedtuple
import numpy as np

import jax.numpy as jnp
from flax import nnx
from tensorflow_probability.substrates.jax.distributions import Categorical, Normal

from scipy.signal import lfilter


ReplayBuffer = namedtuple("ReplayBuffer", "observations actions rewards step_returns")
ExperienceBatch = namedtuple("ExperienceBatch", "obs act ret")


class VPGBuffer(ReplayBuffer):
    def store_step(self, obs, act, rew):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)

    def wrapup_episode(self, len_episode, discount=0.99):
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


class MLPNet(nnx.Module):
    """A simple multi-layer perceptron network"""

    def __init__(
        self,
        rngs: nnx.Rngs,
        input_dims: int,
        output_dims: int,
        hidden_sizes: tuple = (64, 64),
    ):
        self.backbone_sizes = (input_dims, *hidden_sizes)
        self.backbone_transforms = []  # TODO: functionize
        for i in range(len(self.backbone_sizes) - 1):
            self.backbone_transforms.append(
                nnx.Linear(
                    self.backbone_sizes[i], self.backbone_sizes[i + 1], rngs=rngs
                )
            )
        self.output_transform = nnx.Linear(
            self.backbone_sizes[-1], output_dims, rngs=rngs
        )


class CategoricalActor(MLPNet):
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
        pi = Categorical(logits=log_prob)

        return log_prob, pi
