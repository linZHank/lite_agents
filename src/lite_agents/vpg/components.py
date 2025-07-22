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


class CategoricalActor(nnx.Module):
    """Actor for discrete actions space"""

    def __init__(
        self,
        rngs: nnx.Rngs,
        observation_dims: int,
        action_dims: int,
        hidden_sizes: tuple = (64, 64),
    ):
        self.observation_dim = observation_dims
        self.action_dim = action_dims
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
        y = self.output_transform(x)
        log_prob = nnx.log_softmax(y)  # log(pi(a|s))
        pi = Categorical(logits=log_probs)

        return log_prob, pi


class VPGAgent:
    def __init__(
        self,
        env_fn,
        seed: int = 25,
    ) -> None:
        self.actor = CategoricalActor(
            env_fn.observation_space.shape[0], env_fn.action_space.n, seed=seed
        )
        self.rngs = self.actor.rngs

    def make_decision(self, observation: np.ndarray):
        logp, pi = self.actor(observation)
        action = pi.sample(seed=self.rngs)
        # log_prob_a_givens = pi.log_prob(action)
        return action, logp

    @nnx.jit
    def compute_objective(self, experience_batch: ExperienceBatch):
        _, policy_batch = self.actor(experience_batch.obs)
        logpa_batch = policy_batch.log_prob(experience_batch.act)
        objective_batch = experience_batch.ret * logpa_batch  # NOT expected return

        return -objective_batch.mean()




if __name__ == "__main__":
    import gymnasium as gym
    buffer = VPGBuffer([], [], [], [])
    len_episode = 0

    env = gym.make("CartPole-v1", render_mode="rbg_array")
    agent = VPGAgent(env)
    last_obs, info = env.reset()
    for _ in range(env.spec.max_episode_steps):
        act, logp = agent.make_decision(last_obs)
        next_obs, rew, term, trunc, info = env.step(np.array(act))
        print("\n")
        print(f"last observation: {last_obs}")
        print(f"action: {act}")
        print(f"next observation: {next_obs}")
        print(f"reward: {rew}")
        print(f"episode terminated: {term}")
        print(f"episode truncated: {trunc}")
        print(f"info: {info}")
        print("\n")
        buffer.store_step(last_obs, act, rew)
        len_episode += 1
        last_obs = next_obs.copy()
        if term or trunc:
            last_obs, info = env.reset()

    exp_bat = buffer.extract_experience()
    obj_val = agent.compute_objective(exp_bat)


