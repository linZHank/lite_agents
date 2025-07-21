from collections import namedtuple
import numpy as np

import jax.numpy as jnp
from flax import nnx

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
    """A simple fully-connected Neural Network model"""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        input_dim: int,
        output_dim: int,
        hidden_sizes: tuple = (64, 64),
    ):
        assert len(hidden_sizes) >= 1
        self.rngs = rngs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backbone_sizes = (input_dim, *hidden_sizes)
        self.backbone_transforms = []

    def __call__(self, x):
        for i in range(len(self.backbone_sizes) - 1):
            x = nnx.relu(
                nnx.Linear(
                    self.backbone_sizes[i], self.backbone_sizes[i + 1], rngs=self.rngs
                )(x)
            )
        y = nnx.Linear(self.backbone_sizes[-1], self.output_dim, rngs=self.rngs)(x)
        return y


# class VPGAgent:
#     def __init__(self, env_name: str, actor_type: str, seed: int = 25):
#         self.rngs = nnx.Rngs(seed)
#
#     @nnx.jit
#     def make_decision(self, obs: np.ndarray):
#         log_probs = nnx.log_softmax(model(obs))  # policy: log(pi(a|s))
#         distribution = tfp.distributions.Categorical(logits=log_probs)
#         sampled_action = distribution.sample(seed=rngs)
#         # log_prob_as = distribution.log_prob(sampled_action)
#         return sampled_action, log_probs
#
if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="human")
    model = MLPNet(
        rngs=nnx.Rngs(0),
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n,
    )
    dumo = env.observation_space.sample()
    print(model(dumo))
