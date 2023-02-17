import gymnasium as gym
from time import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import haiku as hk 
from lite_agents.dqnax import ReplayBuffer, DQNAgent

key_iter = hk.PRNGSequence(jax.random.PRNGKey(20))
env = gym.make("LunarLander-v2")  # , render_mode="human")
agent = DQNAgent(
    env=env,
    polyak=0.995,
    update_freq=200,
    learning_rate=1e-4,
)
params = agent.init_params(next(key_iter))
agent.init_optimizer(params)
buf = ReplayBuffer(dim_obs=env.observation_space.shape[0], capacity=int(1e6))
episode_count = 0
pobs, info = env.reset(seed=20)
print(f"initial observation: {pobs}, info: {info}")  # debug
term, trun = False, False
rew, episodic_return = 0, 0
deposit_return, averaged_return = [], []
t0 = time()
for step_count in range(200000):
    action, q_val, epsilon = agent.make_decision(
        key=next(key_iter),
        params=params,
        obs=pobs,
        episode_count=episode_count,
        eval_flag=False,
    )
    act = int(action)
    nobs, rew, term, trun, info = env.step(act)
    # print(f"pobs: {nobs}\n act: {act}\n rew: {rew}\n term: {term}\n trun: {trun}\n nobs: {nobs}\n")  # debug
    buf.store(pobs, act, rew, term, nobs)
    episodic_return += rew
    pobs = nobs.copy()
    if buf.is_ready(batch_size=1024):
        online_params, target_params = params.online, params.target
        target_params = agent.update_target_params(online_params, target_params)
        loss_value, params = agent.update_online_params(
            online_params,
            target_params,
            buf.sample(batch_size=1024, discount_rate=0.99),
        )
        # print(f"loss = {loss_value}")
    if term or trun:  # reset env if terminated or truncated
        deposit_return.append(episodic_return)
        averaged_return.append(np.average(deposit_return))
        print(f"episode: {episode_count+1}, step: {step_count+1}, epsilon: {epsilon} \nepisode return: {episodic_return} \nterminated: {term}, truncated: {trun}")
        print(f"averaged_return: {averaged_return[-1]}\n----\n")
        episode_count += 1
        pobs, _ = env.reset(seed=20)
        term, trun = False, False
        rew, episodic_return = 0, 0
t1 = time()
print(f"time consuming: {t1 - t0}")
plt.plot(averaged_return)
plt.show()
