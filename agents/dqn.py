"""Class and functions to implement a simple DQN agent"""

import collections
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
# import rlax


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

    def __init__(self) -> None:
        self.qnet = QNet(num_outputs=4)

    def make_decision(self, key, params, state, episode_count, eval_flag=False):
        """pi(a|s)
        TODO:
            add warm up
            rewrite epsilon greedy w/o rlax
        """
        # state = jnp.expand_dims(state, 0)  # specify batch size
        qvals = jnp.squeeze(self.qnet.apply(params.online, state))
        epsilon = self.epsilon_by_frame(episode_count)
        # sampled_action = rlax.epsilon_greedy(epsilon).sample(key, qvals)
        # greedy_action = rlax.greedy().sample(key, qvals)
        # action = jax.lax.select(eval_flag, greedy_action, sampled_action)

        return action, qvals, epsilon


if __name__=='__main__':
    import gymnasium as gym
    env = gym.make('LunarLander-v2', render_mode='human')
    s_tm1, info = env.reset()
    term, trunc = False, False
    episode_count = 0
    agent = DQNAgent()
    key = jax.random.PRNGKey(20)
    online_params = agent.qnet.init(key, env.observation_space.sample())
    for step in range(1000):
        # a_tm1 = env.action_space.sample()
        new_key, subkey = jax.random.split(key)
        del key
        a_tm1, qvals, epsilon = agent.make_decision(
            key=subkey,
            params=online_params,
            state=s_tm1,
            episode_count=episode_count,
        )
        del subkey
        key = new_key

        s_t, r_t, term, trunc, info = env.step(a_tm1)
        print(f"\n---step: {step}---")
        print(f"state: {s_tm1}\naction: {a_tm1}\nreward: {r_t}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {s_t}")
        s_tm1 = s_t
        if term or trunc:
            s_tm1, info = env.reset()
            term, trunc = False, False
            print("reset")



