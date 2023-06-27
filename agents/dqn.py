"""Class and functions to implement a simple DQN agent"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import rlax


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


class DQNAgent:
    """DQN agent template"""

    def make_decision(self, key, params, state, episode_count, eval_flag=False):
        """pi(a|s)
        TODO:
            add warm up
            rewrite epsilon greedy w/o rlax
        """
        state = jnp.expand_dims(state, 0)  # specify batch size
        q_val = jnp.squeeze(self.qnet.apply(params.online, state))
        epsilon = self.epsilon_by_frame(episode_count)
        sampled_action = rlax.epsilon_greedy(epsilon).sample(key, q_val)
        greedy_action = rlax.greedy().sample(key, q_val)
        action = jax.lax.select(eval_flag, greedy_action, sampled_action)

        return action, q_val, epsilon


if __name__=='__main__':
    import gymnasium as gym
    env = gym.make('LunarLander-v2', render_mode='human')
    s_tm1, info = env.reset()
    term, trunc = False, False
    episode_count = 0
    agent = DQNAgent()
    for step in range(1000):
        a_tm1 = env.action_space.sample()
        a_tm1, q_val, epsilon = agent.make_decision(
            key=next(key_iter),
            params=params,
            state=s_tm1,
            episode_count=episode_count,
        )
        s_t, r_t, term, trunc, info = env.step(a_tm1)
        print(f"\n---step: {step}---")
        print(f"state: {s_tm1}\naction: {a_tm1}\nreward: {r_t}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {s_t}")
        s_tm1 = s_t
        if term or trunc:
            s_tm1, info = env.reset()
            term, trunc = False, False
            print("reset")



