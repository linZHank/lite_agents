from pathlib import Path, PosixPath
from typing import Optional

import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

from spinupax.dqn.components import (
    DQNBuffer,
    ExperienceBatch,
    QValueNet,
)

import matplotlib.pyplot as plt


def learn(
    env_name: str = "CartPole-v1",
    env_options: Optional[dict] = {"render_mode": "rgb_array"},
    seed: int = 25,
    discount: float = 0.99,
    max_steps=int(1e5),
    warmup_episodes: int = 10,
    learning_rate: float = 3e-4,
    hidden_sizes: tuple = (64, 64),
    epsilon_decay_episodes: int = 100,
    experience_sample_size: int = 512,
    eval_flag: bool = False,
    ckpt_dir: PosixPath = Path("/tmp/spinupax/cartpole/dqn/checkpoints/"),
    save_per_epoch: int = 10,
):
    # SETUP
    env = gym.make(env_name, **env_options)
    rngs = nnx.Rngs(seed)
