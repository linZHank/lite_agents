import jax
import jax.numpy as jnp
from lite_agents.dqn import DQNAgent, ReplayBuffer
import gymnasium as gym
import matplotlib.pyplot as plt


# SETUP
env = gym.make('LunarLander-v2')
buffer = ReplayBuffer(int(1e6), env.observation_space.shape)
agent = DQNAgent(
    seed=0,
    obs_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
    hidden_sizes=(256, 256),
    lr=5e-5,
)
print(
    agent.qnet.tabulate(
        jax.random.key(0),
        env.observation_space.sample(),
        compute_flops=True,
        compute_vjp_flops=True
    )
)  # view QNet structure in a table

# LOOP
ep, st, g = 0, 0, 0  # episode_count, episodic_return
deposit_return, average_return = [], []
o_0, i = env.reset()
for _ in range(int(2e5)):
    a = agent.make_decision(jnp.expand_dims(o_0, axis=0), ep, eval_flag=False)
    # print(a)
    o_1, r, t, tr, i = env.step(int(a))
    buffer.store(o_0, a, o_1, r, t)
    g += r
    # print(f"\n---episode: {ep+1}, step: {st+1}, return: {g}---")
    # print(f"state: {o_0}\naction: {a}\nreward: {r}\nterminated? {t}\ntruncated? {tr}\ninfo: {i}\nnext state: {o_1}")
    o_0 = o_1.copy()
    if ep > 10:
        replay = buffer.sample(1024)
        loss = agent.train_fn(replay)
    st += 1
    if t or tr:
        print(f"\n---episode: {ep+1}, steps: {st+1}, return: {g}---\n")
        deposit_return.append(g)
        average_return.append(sum(deposit_return) / len(deposit_return))
        ep += 1
        st = 0
        g = 0
        o_0, i = env.reset()
env.close()
plt.plot(average_return)
plt.show()

# validation
env = gym.make('LunarLander-v2', render_mode='human')
o_0, _ = env.reset()
term, trunc = False, False
for _ in range(1000):
    a = agent.make_decision(jnp.expand_dims(o_0, axis=0), ep)
    o_1, r, t, tr, i = env.step(int(a))
    print(f"state: {o_0}\naction: {a}\nreward: {r}\nterminated? {t}\ntruncated? {tr}\ninfo: {i}\nnext state: {o_1}")
    o_0 = o_1.copy()
    if t or tr:
        break
env.close()

