"""Class and functions to implement a simple DQN agent"""

import collections
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from distrax import Greedy, EpsilonGreedy


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


Params = collections.namedtuple('Params', 'online, target')
class DQNAgent:
    """DQN agent template"""

    def __init__(self, observation_shape, num_actions) -> None:
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.qnet = QNet(num_outputs=num_actions)

    def init_params(self, key, sample_obs):
        online_params = self.qnet.init(key, sample_obs)

        return Params(online_params, online_params)

    def make_decision(self, key, params, state, episode_count, eval_flag=False):
        """pi(a|s)
        TODO:
            add warm up
            rewrite epsilon greedy w/o rlax
        """
        # state = jnp.expand_dims(state, 0)  # specify batch size
        key, subkey = jax.random.split(key)
        qvals = jnp.squeeze(self.qnet.apply(params.online, state))
        epsilon = self.epsilon_by_frame(episode_count)
        sampled_action = EpsilonGreedy(preferences=qvals, epsilon=epsilon).sample(subkey, qvals)
        greedy_action = Greedy(preferences=qvals).sample(subkey, qvals)
        action = jax.lax.select(eval_flag, greedy_action, sampled_action)

        return action, qvals, epsilon


if __name__=='__main__':
    import gymnasium as gym
    env = gym.make('LunarLander-v2')  # , render_mode='human')
    s_tm1, info = env.reset()
    term, trunc = False, False
    episode_count = 0
    agent = DQNAgent(
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n
    )
    key = jax.random.PRNGKey(20)
    params = agent.init_params(
        key=key,
        sample_obs=env.observation_space.sample()
    )
    # for step in range(1000):
    #     # a_tm1 = env.action_space.sample()
    #     a_tm1, qvals, epsilon = agent.make_decision(
    #         key=subkey,
    #         params=online_params,
    #         state=s_tm1,
    #         episode_count=episode_count,
    #     )
    #
    #     s_t, r_t, term, trunc, info = env.step(a_tm1)
    #     print(f"\n---step: {step}---")
    #     print(f"state: {s_tm1}\naction: {a_tm1}\nreward: {r_t}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {s_t}")
    #     s_tm1 = s_t
    #     if term or trunc:
    #         s_tm1, info = env.reset()
    #         term, trunc = False, False
    #         print("reset")
    #
