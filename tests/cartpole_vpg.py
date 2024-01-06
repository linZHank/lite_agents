import gymnasium as gym
from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from distrax import Categorical
import matplotlib.pyplot as plt
from scipy.signal import lfilter


Replay = namedtuple('Replay', ['obs', 'act', 'logp', 'ret'])

class OnPolicyReplayBuffer(object):
    """A simple on-policy replay buffer."""

    def __init__(self, capacity: int, obs_shape: tuple, act_shape: tuple, num_act=None):
        # Variables
        self.loc = 0  # buffer instance index
        self.ep_init_loc = 0  # episode initial index
        # Properties
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.num_act = num_act
        # Replay storages
        self.buf_obs = np.zeros(shape=[capacity]+list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=(capacity, 1), dtype=int)
        self.buf_rews = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_logpas = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_rets = np.zeros_like(self.buf_rews)

    def store(self, obs, action, reward, logp):
        assert self.loc < self.capacity
        self.buf_obs[self.loc] = obs
        self.buf_acts[self.loc] = action
        self.buf_rews[self.loc] = reward
        self.buf_logpas[self.loc] = logp
        self.loc += 1

    def finish_episode(self, discount=0.9):
        """ End of episode process
        Call this at the end of a trajectory, to compute the rewards-to-go.
        """
        print(f"episode srart index: {self.ep_init_loc}")
        ep_slice = slice(self.ep_init_loc, self.loc)
        self.buf_rets[ep_slice] = lfilter([1], [1, -discount], self.buf_rews[ep_slice][::-1], axis=0)[::-1]  # rewards to go
        self.ep_init_loc = self.loc
        print(f"current index: {self.loc}")

    def extract(self):
        """Get replay experience
        """
        replay = Replay(
            self.buf_obs,
            self.buf_acts,
            self.buf_logpas,
            self.buf_rets,
        )
        # self.__init__(self.capacity, self.obs_shape, self.act_shape, self.num_act)
        return replay

class MLP(nn.Module):
    num_outputs: int
    hidden_sizes: tuple = (64, 64)

    @nn.compact
    def __call__(self, inputs):
        dtype = jnp.float32
        x = inputs.astype(dtype)
        for i, size in enumerate(self.hidden_sizes):
            z = nn.Dense(features=size, name='hidden'+str(i+1), dtype=dtype)(x)
            x = nn.relu(z)
        logits = nn.Dense(features=self.num_outputs, name='logits')(x)
        return logits

def make_decision(key, params, obs):
    logits = policy_net.apply(params, obs).squeeze(axis=0)
    distribution = Categorical(logits=logits)
    act = distribution.sample(seed=key)
    logp_a = distribution.log_prob(act)
    return act, logp_a

@jax.jit
def loss_fn(params, data_obs, data_acts, data_rets):
    logits = policy_net.apply(params, data_obs)
    distributions = Categorical(logits=logits)
    logpas = distributions.log_prob(data_acts)
    return -(logpas * data_rets).mean()

# SETUP
key = jax.random.PRNGKey(19)
env = gym.make('CartPole-v1')
buf = OnPolicyReplayBuffer(capacity=500, obs_shape=env.observation_space.shape, act_shape=env.action_space.shape, num_act=env.action_space.n)
policy_net = MLP(env.action_space.n)
params = policy_net.init(
    key,
    jnp.expand_dims(env.observation_space.sample(), axis=0)
)


# LOOP
ep, ep_return = 0, 0
deposit_return, average_return = [], []
pobs, _ = env.reset()
key, subkey = jax.random.split(key)
for st in range(500):
    key, subkey = jax.random.split(key)
    act, logp = make_decision(
        subkey,
        params,
        jnp.expand_dims(pobs, axis=0),
    )
    # print(act, logp_a)
    # act = env.action_space.sample()
    nobs, rew, term, trunc, _ = env.step(int(act))
    buf.store(pobs, act, rew, logp)
    ep_return += rew
    pobs = nobs
    if term or trunc:
        buf.finish_episode()
        deposit_return.append(ep_return)
        average_return.append(sum(deposit_return) / len(deposit_return))
        print(f"\n---episode: {ep+1}, steps: {st+1}, return: {ep_return}---\n")
        ep += 1
        ep_return = 0
        pobs, _ = env.reset()
buf.finish_episode()
rep = buf.extract()
loss_val = loss_fn(params, rep.obs, rep.act, rep.ret)
print(f"loss: {loss_val}")
env.close()
plt.plot(average_return)
plt.show()

# validation
# env = gym.make('CartPole-v1', render_mode='human')
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
#
