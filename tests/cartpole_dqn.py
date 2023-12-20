from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import gymnasium as gym
import matplotlib.pyplot as plt
import optax
from distrax import Greedy, EpsilonGreedy


Batch = namedtuple('Batch', ['pobs', 'act', 'nobs', 'rew', 'term'])
Params = namedtuple('Params', ['online', 'stable'])


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity, obs_shape):
        self.buf_pobs = np.zeros(shape=[capacity]+list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=(capacity, 1), dtype=int)
        self.buf_rews = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_nobs = np.zeros_like(self.buf_pobs)
        self.buf_terms = np.zeros(shape=(capacity, 1), dtype=np.float32)
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

    def sample(self, key, batch_size):
        """Sample a batch of experience
        TODO: try jit
        """
        ids = jax.random.randint(
            key=key,
            shape=(min(self.buffer_size, batch_size),),
            minval=0,
            maxval=self.buffer_size,
        )
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
env = gym.make('CartPole-v1')
buffer = ReplayBuffer(int(1e4), env.observation_space.shape)
qnet = MLP(num_outputs=env.action_space.n, hidden_sizes=(3,))
print(
    qnet.tabulate(
        key,
        env.observation_space.sample(),
        compute_flops=True,
        compute_vjp_flops=True
    )
)  # view QNet structure in a table
online_p = qnet.init(
    key,
    jnp.expand_dims(env.observation_space.sample(), axis=0)
)['params']  # init online parameters
params = Params(online_p, online_p)  # init online and stable parameters
tx = optax.adam(1e-4)
state = train_state.TrainState.create(
    apply_fn=qnet.apply,
    params=params.online,
    tx=tx,
)
epsilon_schedule = optax.linear_schedule(
    init_value=1.0,
    end_value=0.01,
    transition_steps=100,
    transition_begin=10,
)


def make_decision(key, state, obs, epsilon_schedule, episode_counts, eval_flag=True):
    qvalues = state.apply_fn({'params': state.params}, obs).squeeze(axis=0)
    key, subkey = jax.random.split(key)
    epsilon = epsilon_schedule(episode_counts)
    distr_greedy = Greedy(preferences=qvalues).sample(seed=subkey)
    distr_sample = EpsilonGreedy(
        preferences=qvalues,
        epsilon=epsilon).sample(seed=subkey)
    action = jax.lax.select(
        pred=eval_flag,
        on_true=distr_greedy,
        on_false=distr_sample,
    )

    return key, epsilon, qvalues, action


# @jax.vmap
# def double_q_error(batch, q_pred, q_next, q_duel, gamma):
#     q_target = jax.lax.stop_gradient(
#         batch.rew + (1 - batch.term) * gamma * q_next[q_duel.argmax(axis=-1)]
#     )
#     td_error = q_target - q_pred[act]
#     return td_error
#
#
# @jax.jit
# def train_step(state, params, batch):
#     params_stable = optax.incremental_update(
#         new_tensors=params.online,
#         old_tensors=params.stable,
#         step_size=0.01
#     )
#
#     def double_q_loss(params_online, params_stable, discount=0.99):
#         qval_pred = state.apply_fn({'params': params_online}, batch.pob)
#         qval_next = jax.lax.stop_gradient(state.apply_fn({'params': params_stable}, batch.nob))
#         qval_duel = state.apply_fn({'params': params_online}, batch.nob)
#         qerr = double_q_error(
#             batch,
#             discount * jnp.ones_like(batch.rew),
#             qval_pred,
#             qval_next,
#             qval_duel,
#         )
#         loss_value = optax.l2_loss(qerr).mean()
#         return loss_value
#     grad_fn = jax.value_and_grad(double_q_loss)
#     qloss, grads = grad_fn(state.params, params_stable)
#     state = state.apply_gradients(grads=grads)
#     params = Params(params_online, params_stable)
#     return state, qloss, params
#
#
# # LOOP
ep, ep_return = 0, 0  # episode_count, episodic_return
deposit_return, average_return = [], []
pobs, _ = env.reset()
for st in range(300):
    key, subkey = jax.random.split(key)
    key, epsilon, qvals, act = make_decision(
        subkey,
        state,
        jnp.expand_dims(pobs, axis=0),
        epsilon_schedule,
        ep+1,
        eval_flag=False,
    )
    # act = env.action_space.sample()
    # print(act, qvals)
    nobs, rew, term, trunc, _ = env.step(int(act))
    buffer.store(pobs, act, nobs, rew, term)
    ep_return += rew
    # print(f"previous observation: {pobs}")
    # print(f"action: {act}")
    # print(f"next observation: {nobs}")
    # print(f"reward: {rew}")
    # print(f"termination flag: {term}")
    # print(f"truncated flag: {trunc}")
    pobs = nobs.copy()
    if ep >= 1:
        rep = buffer.sample(subkey, 1024)
        # loss, state = train_step(state, params.stable, replay)
    if term or trunc:
        print(f"\n---episode: {ep+1}, epsilon: {epsilon}, steps: {st+1}, return: {ep_return}---\n")
        deposit_return.append(ep_return)
        average_return.append(sum(deposit_return) / len(deposit_return))
        ep += 1
        ep_return = 0
        pobs, _ = env.reset()
env.close()
plt.plot(average_return)
plt.show()

# validation
# env = gym.make('LunarLander-v2', render_mode='human')
# o_0, _ = env.reset()
# term, trunc = False, False
# for _ in range(1000):
#     a = agent.make_decision(jnp.expand_dims(o_0, axis=0), ep)
#     o_1, r, t, tr, i = env.step(int(a))
#     print(f"state: {o_0}\naction: {a}\nreward: {r}\nterminated? {t}\ntruncated? {tr}\ninfo: {i}\nnext state: {o_1}")
#     o_0 = o_1.copy()
#     if t or tr:
#         break
# env.close()
#
