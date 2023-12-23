from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
# from flax.training import train_state
import gymnasium as gym
import matplotlib.pyplot as plt
import optax
from distrax import Greedy, EpsilonGreedy


Batch = namedtuple('Batch', ['pobs', 'act', 'nobs', 'rew', 'term'])
Params = namedtuple('Params', ['online', 'stable'])


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity, obs_shape):
        # Variables
        self.loc = 0  # replay instance index
        self.occupied_size = 0
        # Replay storages
        self.buf_pobs = np.zeros(shape=[capacity]+list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=(capacity, 1), dtype=int)
        self.buf_rews = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_nobs = np.zeros_like(self.buf_pobs)
        self.buf_terms = np.zeros(shape=(capacity, 1), dtype=np.float32)
        # Properties
        self.capacity = capacity

    def store(self, prev_obs, action, next_obs, reward, term_flag):
        self.buf_pobs[self.loc] = prev_obs
        self.buf_acts[self.loc] = action
        self.buf_nobs[self.loc] = next_obs
        self.buf_rews[self.loc] = reward
        self.buf_terms[self.loc] = term_flag
        self.loc = (self.loc + 1) % self.capacity
        self.occupied_size = min(self.occupied_size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of experience
        """
        # if batch_size > self.occupied_size:
        #     print(f"WARNING: batch size {batch_size} > stored size: {self.occupied_size}")
        ids = np.random.randint(
            low=0,
            high=self.occupied_size,
            size=((batch_size,))
        )
        self.sample_ids = ids
        sampled_pobs = self.buf_pobs[ids]
        sampled_acts = self.buf_acts[ids]
        sampled_nobs = self.buf_nobs[ids]
        sampled_rews = self.buf_rews[ids]
        sampled_terms = self.buf_terms[ids]
        sampled_batch = Batch(
            sampled_pobs,
            sampled_acts,
            sampled_nobs,
            sampled_rews,
            sampled_terms,
        )
        return sampled_batch


class MLP(nn.Module):
    """Multi-Layer Perceptron model

    """

    num_outputs: int
    hidden_sizes: tuple

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


# SETUP
key = jax.random.PRNGKey(19)
# tabulate_key, init_params_key, make_decision_key = jax.random.split(key, num=3)
env = gym.make('CartPole-v1')
buffer = ReplayBuffer(int(1e4), env.observation_space.shape)
qnet = MLP(num_outputs=env.action_space.n, hidden_sizes=(128, 128))
print(
    qnet.tabulate(
        key,
        env.observation_space.sample(),
        compute_flops=True,
        compute_vjp_flops=True
    )
)  # view QNet structure in a table
online_parameters = qnet.init(
    key,
    jnp.expand_dims(env.observation_space.sample(), axis=0)
)  # ['params']
params = Params(online_parameters, online_parameters)
epsilon_schedule = optax.linear_schedule(
    init_value=1.0,
    end_value=0.01,
    transition_steps=100,
    transition_begin=10,
)
optimizer = optax.adam(1e-4)
opt_state = optimizer.init(params.online)


def make_decision(key, obs, params, epsilon, eval_flag=True):
    qvals = qnet.apply(params.online, obs).squeeze(axis=0)
    act_greedy = Greedy(preferences=qvals).sample(seed=key)
    act_sample = EpsilonGreedy(
        preferences=qvals,
        epsilon=epsilon).sample(seed=key)
    action = jax.lax.select(
        pred=eval_flag,
        on_true=act_greedy,
        on_false=act_sample,
    )

    return action, qvals


@jax.vmap
def double_q_error(act, rew, term, gamma, q_pred, q_next, q_duel):
    q_target = jax.lax.stop_gradient(
        rew + (1 - term) * gamma * q_next[q_duel.argmax(axis=-1)]
    )
    td_error = q_target - q_pred[act]
    return td_error


@jax.jit
def loss_fn(params_online, params_stable, batch, discount=0.9):
    qval_pred = qnet.apply(params_online, batch.pobs)
    qval_next = qnet.apply(params_stable, batch.nobs)
    qval_duel = qnet.apply(params_online, batch.nobs)
    qerr = double_q_error(
        batch.act,
        batch.rew,
        batch.term,
        discount * jnp.ones_like(batch.rew),
        qval_pred,
        qval_next,
        qval_duel,
    )
    loss_value = optax.l2_loss(qerr).mean()
    return loss_value


@jax.jit
def train_step(params_online, params_stable, batch, opt_state):
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(params_online, params_stable, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params_online = optax.apply_updates(params_online, updates)
    params = Params(params_online, params_stable)
    return params, loss_val, opt_state


@jax.jit
def polyak_update(params):
    params_stable = optax.incremental_update(
        new_tensors=params.online,
        old_tensors=params.stable,
        step_size=0.01
    )
    params = Params(params.online, params_stable)
    return params


# LOOP
ep, ep_return = 0, 0
deposit_return, average_return = [], []
pobs, _ = env.reset()
epsilon = epsilon_schedule(ep + 1)
key, subkey = jax.random.split(key)
for st in range(500):
    key, subkey = jax.random.split(key)
    act, qvals = make_decision(
        subkey,
        jnp.expand_dims(pobs, axis=0),
        params,
        epsilon,
        eval_flag=False,
    )
    # print(act, qvals)
    # act = env.action_space.sample()
    nobs, rew, term, trunc, _ = env.step(int(act))
    buffer.store(pobs, act, nobs, rew, term)
    ep_return += rew
    # print(f"previous observation: {pobs}")
    # print(f"action: {act}")
    # print(f"next observation: {nobs}")
    # print(f"reward: {rew}")
    # print(f"termination flag: {term}")
    # print(f"truncated flag: {trunc}")
    pobs = nobs
    if ep >= 10:
        rep = buffer.sample(256)
        qloss_val = loss_fn(params.online, params.stable, rep)
        print(f"loss: {qloss_val}")
        # params, loss_val, opt_state = train_step(params.online, params.stable, rep, opt_state)
        # params = polyak_update(params)
    #     # loss, state = train_step(state, params.stable, replay)
    if term or trunc:
        deposit_return.append(ep_return)
        average_return.append(sum(deposit_return) / len(deposit_return))
        print(f"\n---episode: {ep+1}, steps: {st+1}, epsilon:{epsilon}, return: {ep_return}---\n")
        ep += 1
        ep_return = 0
        pobs, _ = env.reset()
        epsilon = epsilon_schedule(ep + 1)
env.close()
plt.plot(average_return)
plt.show()

# validation
env = gym.make('CartPole-v1', render_mode='human')
pobs, _ = env.reset()
term, trunc = False, False
for _ in range(500):
    key, subkey = jax.random.split(key)
    act, qvals = make_decision(
        subkey,
        jnp.expand_dims(pobs, axis=0),
        params,
        epsilon,
    )
    nobs, rew, term, trunc, _ = env.step(int(act))
    ep_return += rew
    pobs = nobs
    if term or trunc:
        print(f"\n---return: {ep_return}---\n")
        break
env.close()

