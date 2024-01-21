import gymnasium as gym
from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from distrax import Normal
import matplotlib.pyplot as plt
from scipy.signal import lfilter


Replay = namedtuple('Replay', ['obs', 'act', 'ret'])

class ReplayBuffer(object):
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
        self.buf_obs = np.zeros(shape=[capacity]+list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=(capacity, act_shape[0]), dtype=int)
        self.buf_rews = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_rets = np.zeros_like(self.buf_rews)

    def store(self, observation, action, reward):
        assert self.id < self.capacity
        self.buf_obs[self.id] = observation
        self.buf_acts[self.id] = action
        self.buf_rews[self.id] = reward
        self.id += 1

    def finish_episode(self, discount=0.9):
        """ End of episode process
        Call this at the end of a trajectory, to compute the rewards-to-go.
        """
        # print(f"episode srart index: {self.ep_init_id}")
        ep_slice = slice(self.ep_init_id, self.id)
        self.buf_rets[ep_slice] = lfilter([1], [1, -discount], self.buf_rews[ep_slice][::-1], axis=0)[::-1]  # rewards to go
        self.ep_init_id = self.id
        # print(f"current index: {self.id}")

    def extract(self):
        """Get replay experience
        """
        replay = Replay(
            self.buf_obs,
            self.buf_acts,
            self.buf_rets,
        )
        # clean up replay buffer for next epoch
        self.__init__(self.capacity, self.obs_shape, self.act_shape, self.num_act)
        return replay

class GaussianPolicyNet(nn.Module):
    """An MLP fr continuous action space
    """
    dim_acts: int
    hidden_sizes: tuple = (64, 64)

    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32)
        for i, size in enumerate(self.hidden_sizes):
            z = nn.Dense(features=size, name=f'hidden_{i+1}')(x)
            x = nn.relu(z)
        logits_mean = nn.Dense(features=self.dim_acts, name='mean')(x)
        logits_lstd = nn.Dense(features=self.dim_acts, name='log_std')(x)
        return logits_mean, logits_lstd

def make_decision(key, params, obs):
    mu, log_sigma = actor.apply(params, obs)
    mu = mu.squeeze(axis=0)
    sigma = jnp.exp(log_sigma.squeeze(axis=0))
    distribution = Normal(loc=mu, scale=sigma+1e-10)
    act = distribution.sample(seed=key)
    logp_a = distribution.log_prob(act)
    return act, logp_a

@jax.jit
def loss_fn(params, data_obs, data_acts, data_rets):
    mus, log_sigmas = actor.apply(params, data_obs)
    sigmas = jnp.exp(log_sigmas)
    distributions = Normal(loc=mus, scale=sigmas+1e-10)
    logpas = distributions.log_prob(data_acts)  # squeeze actions data
    return -(logpas * data_rets).mean()  # squeeze returns data

@jax.jit
def train_epoch(params, opt_state, data):
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(params, data.obs, data.act, data.ret)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, loss_val, opt_state

# SETUP
key = jax.random.PRNGKey(19)
env = gym.make('MountainCarContinuous-v0')
buf = ReplayBuffer(
    capacity=5000,
    obs_shape=env.observation_space.shape,
    act_shape=env.action_space.shape,
)
actor = GaussianPolicyNet(dim_acts=env.action_space.shape[0])
params = actor.init(
    key,
    jnp.expand_dims(env.observation_space.sample(), axis=0)
)
optimizer = optax.adam(3e-4)
opt_state = optimizer.init(params)


# LOOP
num_epochs = 20
ep, ep_return = 0, 0
deposit_return, average_return = [], []
pobs, _ = env.reset()
key, subkey = jax.random.split(key)
for e in range(num_epochs):
    for st in range(buf.capacity):
        key, subkey = jax.random.split(key)
        act, logp = make_decision(
            subkey,
            params,
            jnp.expand_dims(pobs, axis=0),
        )
        # print(act, logp)
        nobs, rew, term, trunc, _ = env.step(np.array(act))
        buf.store(pobs, act, rew)
        ep_return += rew
        pobs = nobs
        if term or trunc:
            buf.finish_episode()
            deposit_return.append(ep_return)
            average_return.append(sum(deposit_return) / len(deposit_return))
            print(f"episode: {ep+1}, steps: {st+1}, return: {ep_return}")
            ep += 1
            ep_return = 0
            pobs, _ = env.reset()
    buf.finish_episode()
    rep = buf.extract()
    # loss_val = loss_fn(params, rep.obs, rep.act, rep.ret)
    params, loss_val, opt_state = train_epoch(params, opt_state, rep)
    print(f"\n---epoch {e+1} loss: {loss_val}---\n")
env.close()
plt.plot(average_return)
plt.show()

# VALIDATION
env = gym.make('MountainCarContinuous-v0', render_mode='human')
pobs, _ = env.reset()
term, trunc = False, False
for _ in range(1000):
    key, subkey = jax.random.split(key)
    act, qvals = make_decision(
        subkey,
        params,
        jnp.expand_dims(pobs, axis=0),
    )
    nobs, rew, term, trunc, _ = env.step(int(act))
    ep_return += rew
    pobs = nobs
    if term or trunc:
        print(f"\n---return: {ep_return}---\n")
        break
env.close()

