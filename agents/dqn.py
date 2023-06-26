"""Class and functions to implement a simple DQN agent"""

import flax.linen as nn
import jax.numpy as jnp


class CriticNet(nn.Module):
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


if __name__=='__main__':
    import gymnasium as gym
    env = gym.make('LunarLander-v2', render_mode='human')
    s_tm1, info = env.reset()
    term, trunc = False, False
    for step in range(1000):
        a_tm1 = env.action_space.sample()
        s_t, r_t, term, trunc, info = env.step(a_tm1)
        print(f"\n---step: {step}---")
        print(f"state: {s_tm1}\naction: {a_tm1}\nreward: {r_t}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {s_t}")
        s_tm1 = s_t
        if term or trunc:
            s_tm1, info = env.reset()
            term, trunc = False, False
            print("reset")



