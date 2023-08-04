"""Class and functions to implement a simple DQN agent"""

from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from distrax import Greedy, EpsilonGreedy


# Batch = namedtuple('Batch', ['pobs', 'acts', 'rews', 'termsigs', 'nobs'])
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

    def __init__(self, observation_shape, num_actions) -> None:
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.qnet = QNet(num_outputs=num_actions)
        self.epsilon_by_frame = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.01,
            power=1,
            transition_steps=500,
        )
        self.optimizer = optax.adam(learning_rate=3e-4)
        # variables
        self.update_count = 0
        # properties
        self.update_period = 10
        # jit for speed
        self.make_decision = jax.jit(self.make_decision)
        self.update_params = jax.jit(self.update_params)

    def init_params(self, key, sample_obs):
        online_params = self.qnet.init(key, sample_obs)
        params = Params(online_params, online_params)

        return params

    def init_optmizer(self, init_params):
        self.opt_state = self.optimizer.init(init_params)

    def make_decision(self, key, params, obs, episode_count, eval_flag=False):
        """pi(a|s)
        TODO:
            add warm up
            rewrite epsilon greedy w/o rlax
        """
        key, subkey = jax.random.split(key)  # generate a new key, or sampled action won't change
        qvals = jnp.squeeze(self.qnet.apply(params, obs))
        epsilon = self.epsilon_by_frame(episode_count)
        sampled_action = EpsilonGreedy(preferences=qvals, epsilon=epsilon).sample(seed=subkey)
        greedy_action = Greedy(preferences=qvals).sample(seed=subkey)
        action = jax.lax.select(eval_flag, greedy_action, sampled_action)

        return key, action, qvals, epsilon

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
    # setup env, agent and replay buffer
    env = gym.make('LunarLander-v2')  # , render_mode='human')
    agent = DQNAgent(
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n
    )
    buf = ReplayBuffer(capacity=int(200), dim_obs=env.observation_space.shape)
    key = jax.random.PRNGKey(20)
    params = agent.init_params(
        key=key,
        sample_obs=env.observation_space.sample()
    )
    agent.init_optmizer(params.online)
    # init env
    o_tm1, info = env.reset()
    term, trunc = False, False
    episode_count = 0
    for step in range(200):
        key, a_tm1, qvals, epsilon = agent.make_decision(
            key=key,
            params=params.online,
            obs=o_tm1,
            episode_count=episode_count,
        )
        o_t, r_t, term, trunc, info = env.step(int(a_tm1))
        buf.store(prev_obs=o_tm1, action=a_tm1, reward=r_t, term_signal=term, next_obs=o_t)
        print(f"\n---episode: {episode_count}, step: {step}, epsilon: {epsilon}---")
        # print(f"state: {o_tm1}\naction: {a_tm1}\nreward: {r_t}\nterminated? {term}\ntruncated? {trunc}\ninfo: {info}\nnext state: {o_t}")
        o_tm1 = o_t
        if buf.buffer_size > 10:
            batch_loss, params = agent.update_params(params, buf.sample(batch_size=10))
            print(f"loss: {batch_loss}")
        if term or trunc:
            episode_count += 1
            o_tm1, info = env.reset()
            term, trunc = False, False
            print("reset")
