import gymnasium as gym
from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt
from scipy.signal import lfilter


ReplayBuffer = namedtuple("ReplayBuffer", "obs act ret")


class OnPolicyReplayBuffer(object):
    """A simple on-policy replay buffer."""

    def __init__(self, capacity: int, obs_shape: tuple, act_shape: tuple, num_act=None):
        # Variables
        self.id = 0  # buffer instance index
        self.ep_init_id = 0  # episode initial index
        # Properties
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.num_act = num_act
        # Replay storages
        self.buf_obs = np.zeros(shape=[capacity] + list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=(capacity, 1), dtype=int)
        self.buf_rews = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_rets = np.zeros_like(self.buf_rews)

    def store(self, observation, action, reward):
        assert self.id < self.capacity
        self.buf_obs[self.id] = observation
        self.buf_acts[self.id] = action
        self.buf_rews[self.id] = reward
        self.id += 1

    def finish_episode(self, discount=0.9):
        """End of episode process
        Call this at the end of a trajectory, to compute the rewards-to-go.
        """
        # print(f"episode srart index: {self.ep_init_id}")
        ep_slice = slice(self.ep_init_id, self.id)
        self.buf_rets[ep_slice] = lfilter(
            [1], [1, -discount], self.buf_rews[ep_slice][::-1], axis=0
        )[::-1]  # rewards to go
        self.ep_init_id = self.id
        # print(f"current index: {self.id}")

    def extract(self):
        """Get replay experience"""
        replay = Replay(
            self.buf_obs,
            self.buf_acts,
            self.buf_rets,
        )
        # clean up replay buffer for next epoch
        self.__init__(self.capacity, self.obs_shape, self.act_shape, self.num_act)
        return replay


class PolicyNet(nnx.Module):
    """A simple fully-connected Neural Network model"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 128, rngs=rngs)
        self.linear3 = nnx.Linear(128, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))  # 1st layer
        x = nnx.relu(self.linear2(x))  # 1st layer
        y = self.linear3(x)  # 1st layer
        return y


def make_decision(model: PolicyNet, rngs: nnx.Rngs, obs: np.ndarray, act: np.int32):
    logits = nnx.log_softmax(model(obs))
    distribution = tfp.distributions.Categorical(logits=logits)
    act = distribution.sample(seed=rngs)
    logp_a = distribution.log_prob(act)
    return act, logp_a


@jax.jit
def loss_fn(params, data_obs, data_acts, data_rets):
    logits = policy_net.apply(params, data_obs)
    distributions = Categorical(logits=logits)
    logpas = distributions.log_prob(data_acts.squeeze())  # squeeze actions data
    return -(logpas * data_rets.squeeze()).mean()  # squeeze returns data


@jax.jit
def train_epoch(params, opt_state, data):
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(params, data.obs, data.act, data.ret)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, loss_val, opt_state


# SETUP
# key = jax.random.PRNGKey(19)
env = gym.make("CartPole-v1", render_mode="rgb_array")
# buf = OnPolicyReplayBuffer(
#     capacity=500,
#     obs_shape=env.observation_space.shape,
#     act_shape=env.action_space.shape,
#     num_act=env.action_space.n,
# )
actor = PolicyNet(rngs=nnx.Rngs(0))
# params = policy_net.init(key, jnp.expand_dims(env.observation_space.sample(), axis=0))
optimizer = nnx.Optimizer(actor, optax.adamw(3e-4, 0.9))
# opt_state = optimizer.init(params)
#
#
# LOOP
num_epochs = 100
ep, ep_return = 0, 0
deposit_return, average_return = [], []
prev_obs, _ = env.reset()
# key, subkey = jax.random.split(key)
for e in range(num_epochs):
    for st in range(buf.capacity):
        key, subkey = jax.random.split(key)
        act, logp = make_decision(
            subkey,
            params,
            jnp.expand_dims(pobs, axis=0),
        )
#         # print(act, logp_a)
#         # act = env.action_space.sample()
#         nobs, rew, term, trunc, _ = env.step(int(act))
#         buf.store(pobs, act, rew)
#         ep_return += rew
#         pobs = nobs
#         if term or trunc:
#             buf.finish_episode()
#             deposit_return.append(ep_return)
#             average_return.append(sum(deposit_return) / len(deposit_return))
#             print(f"episode: {ep + 1}, steps: {st + 1}, return: {ep_return}")
#             ep += 1
#             ep_return = 0
#             pobs, _ = env.reset()
#     buf.finish_episode()
#     rep = buf.extract()
#     # loss_val = loss_fn(params, rep.obs, rep.act, rep.ret)
#     params, loss_val, opt_state = train_epoch(params, opt_state, rep)
#     print(f"\n---epoch {e + 1} loss: {loss_val}---\n")
# env.close()
# plt.plot(average_return)
# plt.show()
#
# # VALIDATION
# env = gym.make("CartPole-v1", render_mode="human")
# pobs, _ = env.reset()
# term, trunc = False, False
# for _ in range(500):
#     key, subkey = jax.random.split(key)
#     act, qvals = make_decision(
#         subkey,
#         params,
#         jnp.expand_dims(pobs, axis=0),
#     )
#     nobs, rew, term, trunc, _ = env.step(int(act))
#     ep_return += rew
#     pobs = nobs
#     if term or trunc:
#         print(f"\n---return: {ep_return}---\n")
#         break
# env.close()
