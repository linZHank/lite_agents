from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from distrax import Greedy, EpsilonGreedy


Batch = namedtuple('Batch', ['pob', 'act', 'nob', 'rew', 'term'])


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity, obs_shape):
        self.buf_pobs = np.zeros(shape=[capacity]+list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=capacity, dtype=int)
        self.buf_rews = np.zeros(shape=capacity, dtype=np.float32)
        self.buf_nobs = np.zeros_like(self.buf_pobs)
        self.buf_terms = np.zeros(shape=capacity, dtype=np.float32)
        # variables
        self.loc = 0  # replay instance index
        self.buffer_size = 0
        # property
        self.capacity = capacity

    def store(self, prev_obs, action, next_obs, reward, term_flag):
        self.buf_pobs[self.loc] = prev_obs
        self.buf_acts[self.loc] = action
        self.buf_nobs[self.loc] = next_obs
        self.buf_rews[self.loc] = reward
        self.buf_terms[self.loc] = term_flag
        self.loc = (self.loc + 1) % self.capacity
        self.buffer_size = min(self.buffer_size + 1, self.capacity)

    def sample(self, batch_size, discount_factor=0.99):
        ids = np.random.randint(low=0, high=self.buffer_size, size=(batch_size,))
        self.sample_ids = ids
        sampled_pobs = jnp.array(self.buf_pobs[ids])
        sampled_acts = jnp.array(self.buf_acts[ids])
        sampled_nobs = jnp.array(self.buf_nobs[ids])
        sampled_rews = jnp.array(self.buf_rews[ids])
        sampled_terms = jnp.array(self.buf_terms[ids])
        # sampled_drews = sampled_rews * (1 - sampled_terms) * discount_factor  # BIG MISTAKE HERE
        sampled_batch = Batch(
            sampled_pobs,
            sampled_acts,
            sampled_nobs,
            sampled_rews,  # discounted rewards
            sampled_terms,  # terminal flags
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
        x = inputs.astype(dtype)
        for i, size in enumerate(self.hidden_sizes):
            z = nn.Dense(features=size, name='hidden'+str(i+1), dtype=dtype)(x)
            x = nn.relu(z)
        logits = nn.Dense(features=self.num_outputs, name='logits')(x)
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
        self.qvalue_fn = jax.jit(self.state.apply_fn)

    def update_params(self, replay_batch):
        if self.polyak_step_size > 0:
            self.params_stable = optax.incremental_update(
                new_tensors=self.params_online,
                old_tensors=self.params_stable,
                step_size=self.polyak_step_size,
            )
        # else:
        #     self.params_stable = optax.periodic_update(
        #         new_tensors=self.params_online,
        #         old_tensors=self.params_stable,
        #         step=self.update_step,
        #         update_period=self.update_period
        #     )
        loss, grads = jax.value_and_grad(self.loss_fn)(
            self.params_online, self.params_stable, replay_batch
        )


    def make_decision(self, obs, episode_count, eval_flag=True):
        qvals = self.qvalue_fn({'params': self.params_online}, obs).squeeze(axis=0)
        self.key, subkey = jax.random.split(self.key)
        epsilon = self.epsilon_schedule(episode_count)
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
    import matplotlib.pyplot as plt
    # SETUP
    env = gym.make('CartPole-v1', render_mode='human')
    buffer = ReplayBuffer(10000, env.observation_space.shape)
    agent = DQNAgent(
        seed=0,
        obs_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        hidden_sizes=(5, 3),
    )
    print(
        agent.qnet.tabulate(
            jax.random.key(0),
            env.observation_space.sample(),
            compute_flops=True,
            compute_vjp_flops=True
        )
    )  # view QNet structure in a table
    ep, st, g = 0, 0, 0  # episode_count, episodic_return
    deposit_return, average_return = [], []

    # LOOP
    o_0, i = env.reset()
    for _ in range(1000):
        a = agent.make_decision(jnp.expand_dims(o_0, axis=0), 1, eval_flag=False)
        # print(a)
        o_1, r, t, tr, i = env.step(int(a))
        buffer.store(o_0, a, o_1, r, t)
        g += r
        print(f"\n---episode: {ep+1}, step: {st+1}, return: {g}---")
        print(f"state: {o_0}\naction: {a}\nreward: {r}\nterminated? {t}\ntruncated? {tr}\ninfo: {i}\nnext state: {o_1}")
        o_0 = o_1.copy()
        st += 1
        if t or tr:
            deposit_return.append(g)
            average_return.append(sum(deposit_return) / len(deposit_return))
            ep += 1
            st = 0
            g = 0
            o_0, i = env.reset()
            if ep >= 2:
                break
    env.close()
    plt.plot(average_return)
    plt.show()


