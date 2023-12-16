from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from distrax import Greedy, EpsilonGreedy


Batch = namedtuple('Batch', ['pobs', 'acts', 'discrews', 'nobs'])


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity, obs_shape):
        self.buf_pobs = np.zeros(shape=[capacity]+list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=capacity, dtype=int)
        self.buf_rews = np.zeros(shape=capacity, dtype=np.float32)
        self.buf_nobs = np.zeros_like(self.buf_pobs)
        self.buf_terms = np.zeros(shape=capacity, dtype=np.float32)
        self.Batch = namedtuple('Batch', 'pobs, acts, drews, nobs')
        # variables
        self.loc = 0  # replay instance index
        self.buffer_size = 0
        # property
        self.capacity = capacity

    def store(self, prev_obs, action, reward, next_obs, term_flag):
        self.buf_pobs[self.loc] = prev_obs
        self.buf_acts[self.loc] = action
        self.buf_rews[self.loc] = reward
        self.buf_nobs[self.loc] = next_obs
        self.buf_terms[self.loc] = term_flag
        self.loc = (self.loc + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self, batch_size, discount_factor=0.99):
        ids = np.random.randint(low=0, high=self.buffer_size, size=(batch_size,))
        samples_pobs = jnp.array(self.buf_pobs[ids])
        samples_acts = jnp.array(self.buf_acts[ids])
        samples_rews = jnp.array(self.buf_rews[ids])
        samples_nobs = jnp.array(self.buf_nobs[ids])
        samples_terms = jnp.array(self.buf_terms[ids])
        samples_drews = samples_rews * (1 - samples_terms) * discount_factor
        sampled_batch = self.Batch(
            samples_pobs,
            samples_acts,
            # rews_samples,
            # termsigs_samples,
            samples_drews,  # discounted rewards
            samples_nobs
        )
        return sampled_batch

    def is_ready(self, batch_size):  # warm up trick
        return batch_size <= self.capacity


class MLP(nn.Module):
    """Multi-Layer Perceptron model

    """

    num_outputs: int
    hidden_sizes: list

    @nn.compact
    def __call__(self, inputs):
        """Forward pass

        """
        dtype = jnp.float32
        X = inputs.astype(dtype)
        for i, size in enumerate(self.hidden_sizes):
            Z = nn.Dense(features=size, name='hidden'+str(i+1), dtype=dtype)(X)
            X = nn.relu(Z)
        logits = nn.Dense(features=self.num_outputs, name='logits')(X)
        return logits


class DQNAgent:
    """RL agent powered by Deep-Q Network

    """
    def __init__(self, seed, obs_shape, num_actions, hidden_sizes,):
        self.key = jax.random.PRNGKey(seed)
        self.qnet = MLP(num_actions, hidden_sizes)
        self.params_online = self.qnet.init(
            self.key,
            jnp.expand_dims(jnp.ones(obs_shape), axis=0)
        )['params']
        # self.params_target = self.params_online.copy()
        self.epsilon_schedule = optax.linear_schedule(
            init_value=1.0,
            end_value=0.01,
            transition_steps=100,
            transition_begin=10
        )
        # self.lr_schedule = optax.linear_schedule(
        #     init_value=3e-4,
        #     end_value=1e-4,
        #     transition_steps=10000,
        # )
        self.lr = 1e-4
        self.tx = optax.adam(self.lr)
        self.state = train_state.TrainState.create(
          apply_fn=self.qnet.apply,
          params=self.params_online,
          tx=self.tx,
        )

        # Jitted methods
        self.value_fn = jax.jit(self.state.apply_fn)

    def make_decision(self, obs, episode_count, eval_flag=True):
        self.key, subkey = jax.random.split(self.key)
        epsilon = self.epsilon_schedule(episode_count)
        qvals = self.value_fn({'params': self.params_online}, obs).squeeze(axis=0)
        act_greedy = Greedy(preferences=qvals).sample(seed=subkey)
        act_sample = EpsilonGreedy(
            preferences=qvals,
            epsilon=epsilon).sample(seed=subkey)
        action = jax.lax.select(
            pred=eval_flag,
            on_true=act_greedy,
            on_false=act_sample,
        )

        return action


if __name__ == '__main__':
    import gymnasium as gym
    # Create env, agent
    env = gym.make('CartPole-v1', render_mode='human')
    agent = DQNAgent(
        0,
        env.observation_space.shape,
        env.action_space.n,
        (5, 3),
    )
    print(
        agent.qnet.tabulate(
            jax.random.key(0),
            env.observation_space.sample(),
            compute_flops=True,
            compute_vjp_flops=True
        )
    )  # log network details
    # Train agent
    o, i = env.reset()
    for _ in range(1000):
        a = agent.make_decision(jnp.expand_dims(o, axis=0), 1, eval_flag=False)
        print(a)
        o, r, t, tr, i = env.step(int(a))
        if t:
            break
    env.close()


