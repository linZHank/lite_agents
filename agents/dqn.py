"""Class and functions to implement a simple DQN agent"""

from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from distrax import Greedy, EpsilonGreedy


Batch = namedtuple('Batch', ['pobs', 'acts', 'discrews', 'nobs'])
Params = namedtuple('Params', 'online, target')


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity, dim_obs):
        self.pobs_buf = np.zeros(shape=[capacity]+list(dim_obs), dtype=np.float32)
        self.acts_buf = np.zeros(shape=capacity, dtype=int)
        self.rews_buf = np.zeros(shape=capacity, dtype=np.float32)
        self.nobs_buf = np.zeros_like(self.pobs_buf)
        self.termsigs_buf = np.zeros(shape=capacity, dtype=np.float32)
        # variables
        self.loc = 0  # replay instance index
        self.buffer_size = 0
        # property
        self.capacity = capacity

    def store(self, prev_obs, action, reward, term_signal, next_obs):
        self.pobs_buf[self.loc] = prev_obs
        self.acts_buf[self.loc] = action
        self.rews_buf[self.loc] = reward
        self.nobs_buf[self.loc] = next_obs
        self.termsigs_buf[self.loc] = term_signal
        self.loc = (self.loc + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self, batch_size, discount_factor=0.99):
        indices = np.random.randint(low=0, high=self.buffer_size, size=(batch_size,))
        pobs_samples = jnp.asarray(self.pobs_buf[indices])
        nobs_samples = jnp.asarray(self.nobs_buf[indices])
        acts_samples = jnp.asarray(self.acts_buf[indices])
        rews_samples = jnp.asarray(self.rews_buf[indices])
        termsigs_samples = jnp.asarray(self.termsigs_buf[indices])
        drews_samples = rews_samples * (1 - termsigs_samples) * discount_factor
        sampled_batch = Batch(
            pobs_samples,
            acts_samples,
            # rews_samples,
            # termsigs_samples,
            drews_samples,  # discounted rewards
            nobs_samples
        )
        return sampled_batch

    def is_ready(self, batch_size):  # warm up trick
        return batch_size <= self.capacity


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

    def __init__(
        self,
        observation_shape: tuple,
        num_actions: int,
        key: int = 123,
        epsilon_transition_episodes: int = 500,
        learning_rate: float = 1e-4,
        target_update_period: int = 100,
    ):
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.qnet = QNet(num_outputs=num_actions)
        self.epsilon_by_frame = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.01,
            power=1,
            transition_steps=epsilon_transition_episodes,
        )
        self.optimizer = optax.adam(learning_rate=learning_rate)
        # variables
        self.update_count = 0
        # properties
        self.key = jax.random.PRNGKey(key)
        self.update_period = target_update_period
        # jit for speed
        self.make_decision = jax.jit(self.make_decision)
        self.update_params = jax.jit(self.update_params)

    def init_params(self, sample_obs):
        online_params = self.qnet.init(self.key, sample_obs)
        params = Params(online_params, online_params)

        return params

    def init_optmizer(self, init_params):
        self.opt_state = self.optimizer.init(init_params)

    def make_decision(self, params, obs, episode_count, eval_flag=False):
        """pi(a|s)
        TODO:
            add warm up
        """
        self.key, subkey = jax.random.split(self.key)  # generate a new key, or sampled action won't change
        qvals = jnp.squeeze(self.qnet.apply(params, obs))
        epsilon = self.epsilon_by_frame(episode_count)
        sampled_action = EpsilonGreedy(preferences=qvals, epsilon=epsilon).sample(seed=subkey)
        greedy_action = Greedy(preferences=qvals).sample(seed=subkey)
        action = jax.lax.select(eval_flag, greedy_action, sampled_action)

        return action, qvals, epsilon

    def update_params(self, params, replay_batch):
        """Periodic update online params.

        TODO: add polyak update
        """
        target_params = optax.periodic_update(
            params.online,
            params.target,
            self.update_count,
            self.update_period,
        )
        loss_val, grads = jax.value_and_grad(self.loss_fn)(
            params.online, params.target, replay_batch
        )
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        online_params = optax.apply_updates(params.online, updates)
        self.update_count += 1

        return loss_val, Params(online_params, target_params)

    def loss_fn(
        self,
        online_params,
        target_params,
        replay_batch,
        # discount_rate,
    ):
        def double_q_loss(discrews, acts, pred_q, next_q, duel_q):
            target_q = jax.lax.stop_gradient(
                # rew + (1 - termsig) * gamma * next_q[duel_q.argmax(axis=-1)]
                discrews * next_q[duel_q.argmax(axis=-1)]
            )
            td_error = target_q - pred_q[acts]
            return td_error

        pred_qval = self.qnet.apply(online_params, replay_batch.pobs)
        next_qval = self.qnet.apply(target_params, replay_batch.nobs)
        duel_qval = self.qnet.apply(online_params, replay_batch.nobs)
        # discounted_rews = replay_batch.rews * (1 - replay_batch.termsigs) * discount_rate
        vectorized_double_q_loss = jax.vmap(double_q_loss)
        td_loss = vectorized_double_q_loss(
            # discounted_rews,
            replay_batch.discrews,
            replay_batch.acts,
            pred_qval,
            next_qval,
            duel_qval,
        )
        return optax.l2_loss(td_loss).mean()


if __name__ == '__main__':
    import gymnasium as gym
    import matplotlib.pyplot as plt
    # setup env, agent and replay buffer
    env = gym.make('LunarLander-v2')  # , render_mode='human')
    agent = DQNAgent(
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        key=0,
    )
    buf = ReplayBuffer(capacity=int(1e6), dim_obs=env.observation_space.shape)
    params = agent.init_params(sample_obs=env.observation_space.sample())
    agent.init_optmizer(params.online)
    # init env
    o_tm1, info = env.reset()
    term, trunc = False, False
    episode_count, episodic_return = 0, 0
    deposit_return, averaged_return = [], []
    for step in range(int(3e5)):
        a_tm1, qvals, epsilon = agent.make_decision(
            params=params.online,
            obs=o_tm1,
            episode_count=episode_count,
        )
        o_t, r_t, term, trunc, info = env.step(int(a_tm1))
        buf.store(prev_obs=o_tm1, action=a_tm1, reward=r_t, term_signal=term, next_obs=o_t)
        episodic_return += r_t
        # print(f"\n---episode: {episode_count}, step: {step}, return: {episodic_return}---")
        # print(f"state: {o_tm1}\naction: {a_tm1}\nreward: {r_t}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {o_t}")
        o_tm1 = o_t
        if buf.buffer_size > 2048:
            batch_loss, params = agent.update_params(params, buf.sample(batch_size=2048))
            # print(f"loss: {batch_loss}")
        if term or trunc:
            episode_count += 1
            deposit_return.append(episodic_return)
            averaged_return.append(sum(deposit_return) / episode_count)
            print(f"\n---episode: {episode_count}, steps: {step}, epsilon: {epsilon}, average return: {averaged_return[-1]}---\n")
            o_tm1, info = env.reset()
            term, trunc = False, False
            episodic_return = 0
            print("reset")
    plt.plot(averaged_return)
    plt.show()
    # validation
    env = gym.make('LunarLander-v2', render_mode='human')
    o_tm1, info = env.reset()
    term, trunc = False, False
    for _ in range(1000):
        a_tm1, qvals, epsilon = agent.make_decision(
            params=params.online,
            obs=o_tm1,
            episode_count=episode_count,
            eval_flag=True,
        )
        o_t, r_t, term, trunc, info = env.step(int(a_tm1))
        if term or trunc:
            break


