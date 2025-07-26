from collections import namedtuple

import numpy as np
import jax.numpy as jnp
from flax import nnx
from tensorflow_probability.substrates.jax.distributions import Categorical, Normal

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

    def wrapup_episode(self, eoe_value, episode_len, discount=0.99, compromise=0.97):
        """
        Process raw data after an episode, calculate returns-to-go and GAE advantage estimations.
        Args:
            eoe_value: end of episode value estimation.
            episode_len: number of steps in the just-finished episode.
            discount (gamma): price of a reward in future will be penalized at present.
            compromise (lambda): balance variance and bias of advantage estimation.
                GAE(gamma, lambda=0): r_t + gamma V(s_{t+1}) - V(s_t), high bias low variance
                GAE(gamma, lambda=1): sum_{l=0}^{infty} gamma^l r+{t+1} - V(s_t), low bias high variance
        """
        next_vals = self.values[-episode_len + 1 :]
        next_vals.append(eoe_value)
        nv_arr = jnp.array(next_vals)
        r_arr = jnp.array(self.rewards[-episode_len:])
        v_arr = jnp.array(self.values[-episode_len:])
        # GAE-Lambda advantage
        td_errs = r_arr + discount * nv_arr - v_arr  # r_t + gamma V(s_{t+1}) - V(s_t)
        gae_advs = jnp.flip(
            lfilter([1], [1, -discount * compromise], jnp.flip(td_errs)), axis=0
        )
        self.advantages.extend(gae_advs.tolist())
        # Discounted returns to-go
        rev_ep_rews = self.rewards[-episode_len:][::-1]  # reversed episodic rewards
        drtg = lfilter([1], [1, -discount], rev_ep_rews)[::-1]  # discounted return togo
        self.step_returns.extend(drtg.tolist())

    def extract_experience(self):
        observations_batch = jnp.array(self.observations)
        actions_batch = jnp.array(self.actions)
        # returns_batch = jnp.expand_dims(jnp.array(self.step_returns), axis=-1)
        # advantages_batch = jnp.expand_dims(jnp.array(self.advantages), axis=-1)
        returns_batch = jnp.array(self.step_returns)
        advantages_batch = jnp.array(self.advantages)

        experience_batch = ExperienceBatch(
            observations_batch, actions_batch, returns_batch, advantages_batch
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

        return pi


class GaussianActor(MLPNet):
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


class Critic(MLPNet):
    """Critic Net"""

    def __init__(
        self,
        rngs: nnx.Rngs,
        observation_dims: int,
        value_dims: int = 1,
        hidden_sizes: tuple = (64, 64),
    ):
        super().__init__(
            rngs=rngs,
            input_dims=observation_dims,
            output_dims=value_dims,
            hidden_sizes=hidden_sizes,
        )

    def __call__(self, x):
        for trans in self.backbone_transforms:
            x = nnx.relu(trans(x))
        v = self.output_transform(x)
        return v


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    dummy_buffer = ACBuffer([], [], [], [], [], [])
    lo, i = env.reset()
    for _ in range(1024):
        a = env.action_space.sample()
        no, r, te, tr, i = env.step(a)
        dummy_buffer.store_step(lo, a, r, v)
        lo = no.copy()
        if te or tr:
            lo, i = env.reset()
