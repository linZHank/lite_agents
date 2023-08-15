import jax
from dqn import ReplayBuffer, DQNAgent
import gymnasium as gym
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1')  # , render_mode='human')
agent = DQNAgent(
    obs_shape=env.observation_space.shape,
    num_actions=env.action_space.n,
)
buf = ReplayBuffer(capacity=int(1e5), dim_obs=env.observation_space.shape)
# init params and optimizer
key = jax.random.PRNGKey(123)
qnet_params = agent.init_params(key, env.observation_space.sample())
opt_state = agent.init_optmizer(qnet_params.online)
# init env
pobs, _ = env.reset()
term, trunc = False, False
episode_count, episodic_return = 0, 0
deposit_return, averaged_return = [], []
for step in range(int(1e5)):
    key, act, qvals, epsilon = agent.make_decision(
        key=key,
        params=qnet_params.online,
        obs=pobs,
        episode_count=episode_count,
    )
    nobs, rew, term, trunc, info = env.step(int(act))
    buf.store(prev_obs=pobs, action=act, reward=rew, term_signal=term, next_obs=nobs)
    episodic_return += rew
    # print(f"\n---episode: {episode_count}, step: {step}, return: {episodic_return}---")
    # print(f"state: {pobs}\naction: {act}\nreward: {rew}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {nobs}")
    pobs = nobs
    if buf.buffer_size > 1024:
        batch_loss, qnet_params, opt_state = agent.update_params(qnet_params, buf.sample(batch_size=1024), opt_state)
        # print(f"loss: {batch_loss}")
    if term or trunc:
        episode_count += 1
        deposit_return.append(episodic_return)
        averaged_return.append(sum(deposit_return) / episode_count)
        print(f"\n---episode: {episode_count}, steps: {step}, epsilon: {epsilon}, average return: {averaged_return[-1]}---\n")
        pobs, _ = env.reset()
        term, trunc = False, False
        episodic_return = 0
        print("reset")
# validation
env = gym.make('CartPole-v1', render_mode='human')
pobs, _ = env.reset()
term, trunc = False, False
for _ in range(1000):
    key, act, qvals, epsilon = agent.make_decision(
        key=key,
        params=qnet_params.online,
        obs=pobs,
        episode_count=episode_count,
        eval_flag=True,
    )
    nobs, rew, term, trunc, info = env.step(int(act))
    print(f"state: {pobs}\naction: {act}\nreward: {rew}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {nobs}")
    pobs = nobs
    if term or trunc:
        break
env.close()
plt.plot(averaged_return)
plt.show()
