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
