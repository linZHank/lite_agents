import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
# from flax.training import train_state
import gymnasium as gym
import matplotlib.pyplot as plt
from lite_agents.dqn import ReplayBuffer, DQNAgent
import optax
from distrax import Greedy, EpsilonGreedy


import gymnasium as gym
import matplotlib.pyplot as plt
# SETUP
env = gym.make('LunarLander-v2')
buffer = ReplayBuffer(100000, env.observation_space.shape)
agent = DQNAgent(
    seed=19,
    obs_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    hidden_sizes=(128, 128),
    epsilon_decay_episodes=400,
    warmup_episodes=20,
    polyak_step_size=0.02,
)
print(
    agent.qnet.tabulate(
        agent.key,
        env.observation_space.sample(),
        compute_flops=True,
        compute_vjp_flops=True
    )
)  # view QNet structure in a table
# Initialize agent
params = agent.init_params()
agent.init_optimizer(params)
ep_return = 0
deposit_return, average_return = [], []
pobs, _ = env.reset()


# LOOP
for st in range(200000):
    act, qvals = agent.make_decision(
        jnp.expand_dims(pobs, axis=0),
        params,
        eval_flag=False,
    )
    nobs, rew, term, trunc, _ = env.step(int(act))
    buffer.store(pobs, act, nobs, rew, term)
    ep_return += rew
    if agent.ep_count >= agent.warmup_episodes:
        replay = buffer.sample(1024)
        loss_val, params = agent.train_step(params, replay)
    pobs = nobs
    if term or trunc:
        agent.ep_count += 1
        deposit_return.append(ep_return)
        average_return.append(sum(deposit_return) / len(deposit_return))
        print(f"\n---episode: {agent.ep_count}, steps: {st+1}, epsilon:{agent.epsilon}, return: {ep_return}---\n")
        ep_return = 0
        pobs, _ = env.reset()
env.close()
plt.plot(average_return)
plt.show()

# validation
env = gym.make('LunarLander-v2', render_mode='human')
pobs, _ = env.reset()
term, trunc = False, False
for _ in range(1000):
    act, qvals = agent.make_decision(
        jnp.expand_dims(pobs, axis=0),
        params,
    )
    nobs, rew, term, trunc, _ = env.step(int(act))
    ep_return += rew
    pobs = nobs
    if term or trunc:
        print(f"\n---return: {ep_return}---\n")
        break
env.close()

