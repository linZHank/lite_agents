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

    def make_decision(self, episode_count, obs, eval_flag=True):
        self.key, subkey = jax.random.split(self.key)
        epsilon = self.epsilon_schedule(episode_count)
        qvals = self.value_fn(
            {'params': self.state.params},
            obs
        ).squeeze(axis=0)
        act_g = Greedy(preferences=qvals).sample(seed=subkey)  # greedy action
        act_s = EpsilonGreedy(
            preferences=qvals,
            epsilon=epsilon).sample(seed=subkey)  # sampled action
        action = jax.lax.select(
            pred=eval_flag,
            on_true=act_g,
            on_false=act_s,
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
        a = agent.make_decision(1, jnp.expand_dims(o, axis=0), eval_flag=False)
        print(a)
        o, r, t, tr, i = env.step(int(a))
        if t:
            break
    env.close()


