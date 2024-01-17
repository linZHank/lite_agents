# lite_agents
This project is extremely inspired by [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/).

## GPU [jax](https://github.com/google/jax) Installation
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Installation
```bash
git clone https://github.com/linzhank/lite_agents.git
cd lite_agents
pip install -e .
```

## Example
```python
from lite_agents.dqn import ReplayBuffer, DQNAgent
import gymnasium as gym
import jax.numpy as jnp
import matplotlib.pyplot as plt

# SETUP
env = gym.make('CartPole-v1')
buffer = ReplayBuffer(10000, env.observation_space.shape)
agent = DQNAgent(
    seed=19,
    obs_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    hidden_sizes=(64, 64),
)
print(
    agent.qnet.tabulate(
        agent.key,
        env.observation_space.sample(),
        compute_flops=True,
        compute_vjp_flops=True
    )
)  # view QNet structure in a table
params = agent.init_params()
agent.init_optimizer(params)
ep_return = 0
deposit_return, average_return = [], []
pobs, _ = env.reset()

# TRAIN
for st in range(10000):
    act, qvals = agent.make_decision(
        jnp.expand_dims(pobs, axis=0),
        params,
        eval_flag=False,
    )
    nobs, rew, term, trunc, _ = env.step(int(act))
    buffer.store(pobs, act, nobs, rew, term)
    ep_return += rew
    if agent.ep_count >= agent.warmup_episodes:
        replay = buffer.sample(256)
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

# VALIDATE
env = gym.make('CartPole-v1', render_mode='human')
pobs, _ = env.reset()
term, trunc = False, False
for _ in range(500):
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
```
