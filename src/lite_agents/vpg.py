from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from distrax import Categorical, Normal
from scipy.signal import lfilter


Replay = namedtuple('Replay', ['obs', 'act', 'ret'])

class ReplayBuffer(object):
    """A simple on-policy replay buffer."""

    def __init__(self, capacity: int, obs_shape: tuple, act_shape: tuple, act_n=None):
        # Variables
        self.loc = 0  # buffer instance index
        self.ep_init_loc = 0  # episode initial index
        # Properties
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.act_n = act_n
        # Replay storages
        self.buf_obs = np.zeros(shape=[capacity]+list(self.obs_shape), dtype=np.float32)
        if self.act_n:  # discrete action space
            self.buf_act = np.zeros(shape=(capacity, 1), dtype=np.float32)
        else:
            self.buf_act = np.zeros(shape=[capacity]+list(self.act_shape), dtype=np.float32)
        self.buf_rew = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_ret = np.zeros_like(self.buf_rew)

    def store(self, observation, action, reward):
        assert self.loc < self.capacity
        self.buf_obs[self.loc] = observation
        self.buf_act[self.loc] = action
        self.buf_rew[self.loc] = reward
        self.loc += 1

    def finish_episode(self, discount=0.98):
        """ End of episode process
        Call this at the end of a trajectory, to compute the return-to-go.
        """
        def compute_rtgs(rewards):
            return lfilter([1], [1, -discount], rewards[::-1], axis=0,)[::-1]

        ep_slice = slice(self.ep_init_loc, self.loc)
        self.buf_ret[ep_slice] = compute_rtgs(self.buf_rew[ep_slice])
        self.ep_init_loc = self.loc

    def extract(self):
        """Get all on-policy replay experience
        """
        replay = Replay(self.buf_obs, self.buf_act, self.buf_ret)
        # clean up replay buffer for next epoch
        self.__init__(self.capacity, self.obs_shape, self.act_shape, self.act_n)
        return replay

class CategoricalPolicyNet(nn.Module):
    """An MLP for discrete action space
    """
    act_n: int
    hidden_sizes: tuple

    @nn.compact
    def __call__(self, obs: np.ndarray):
        x = obs.astype(jnp.float32)
        for i, size in enumerate(self.hidden_sizes):
            z = nn.Dense(features=size, name='hidden'+str(i+1))(x)
            x = nn.relu(z)
        logits = nn.Dense(features=self.act_n, name='logits')(x)
        return logits

class GaussianPolicyNet(nn.Module):
    """An MLP for continuous action space
    """
    act_shape: int
    hidden_sizes: tuple = (64, 64)

    @nn.compact
    def __call__(self, obs: np.ndarray):
        x = obs.astype(jnp.float32)
        for i, size in enumerate(self.hidden_sizes):
            z = nn.Dense(features=size, name=f'hidden_{i+1}')(x)
            x = nn.relu(z)
        logits_mean = nn.Dense(features=self.act_shape, name='mean')(x)
        logits_lstd = nn.Dense(features=self.act_shape, name='log_std')(x)
        return logits_mean, logits_lstd

class VPGAgent:
    """On-Policy agent powered by REINFORCE (Vanilla Policy Gradient)

    """
    def __init__(
        self,
        seed,
        env,
        hidden_sizes=(64, 64),
        learning_rate=3e-4,
    ):
        # Properties
        self.key = jax.random.PRNGKey(seed)
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        self.act_n = None
        if not len(self.act_shape):  # discrete
            self.act_n = env.action_space.n
        self.lr = learning_rate
        # Policy network
        if self.act_n:
            self.actor = CategoricalPolicyNet(self.act_n, hidden_sizes)
        else:
            self.actor = GaussianPolicyNet(self.act_shape[0], hidden_sizes)
        # JIT compile
        if self.act_n:
            self.loss_fn = jax.jit(self.categorical_loss_fn)
        else:
            self.loss_fn = jax.jit(self.normal_loss_fn)
        # self.categorical_loss_fn = jax.jit(self.categorical_loss_fn)
        # self.normal_loss_fn = jax.jit(self.normal_loss_fn)
        self.train_epoch = jax.jit(self.train_epoch)


    def init_params(self):
        parameters = self.actor.init(
            self.key,
            jnp.expand_dims(jnp.ones(self.obs_shape), axis=0)
        )
        return parameters

    def init_optimizer(self, params):
        self.optimizer = optax.adam(self.lr)
        self.opt_state = self.optimizer.init(params)

    def make_decision(self, params, obs):
        self.key, subkey = jax.random.split(self.key)
        if self.act_n:
            logits = self.actor.apply(params, obs).squeeze(axis=0)
            distribution = Categorical(logits=logits)
        else:
            logits_mu, logits_logsigma = self.actor.apply(params, obs)
            logits_mu = logits_mu.squeeze(axis=0)
            logits_sigma = jnp.exp(logits_logsigma.squeeze(axis=0))
            distribution = Normal(loc=logits_mu, scale=logits_sigma+1e-10)
        act = distribution.sample(seed=subkey)
        logprob_a = distribution.log_prob(act)
        return act, logprob_a

    def categorical_loss_fn(self, params, replay):
        logits = self.actor.apply(params, replay.obs)
        distributions = Categorical(logits=logits)
        logpas = distributions.log_prob(replay.act.squeeze())  # squeeze actions data
        return -(logpas * replay.ret.squeeze()).mean()  # squeeze returns data

    def normal_loss_fn(self, params, replay):
        logits_mu, logits_logsigma = self.actor.apply(params, replay.obs)
        logits_sigma = jnp.exp(logits_logsigma)
        distributions = Normal(loc=logits_mu, scale=logits_sigma)
        logprob_a_s = distributions.log_prob(replay.act)
        return -(logprob_a_s * replay.ret).mean()  # squeeze returns data

    def train_epoch(self, params, replay):
        loss_grad_fn = jax.value_and_grad(self.loss_fn)
        loss_val, grads = loss_grad_fn(params, replay)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        params = optax.apply_updates(params, updates)
        return params, loss_val 



if __name__=='__main__':
    import gymnasium as gym
    import matplotlib.pyplot as plt
    # SETUP
    key = jax.random.PRNGKey(19)
    # env = gym.make('CartPole-v1')
    # env = gym.make('Pendulum-v1', g=9.81)
    env = gym.make('LunarLander-v2', continuous=True)
    buffer = ReplayBuffer(
        capacity=4096,
        obs_shape=env.observation_space.shape,
        act_shape=env.action_space.shape,
        # act_n=env.action_space.n,
    )
    agent = VPGAgent(
        seed=19,
        env=env,
        hidden_sizes=(128, 128),
    )
    params = agent.init_params()
    agent.init_optimizer(params)

    num_epochs = 100
    ep, ep_return = 0, 0
    deposit_return, average_return = [], []
    pobs, _ = env.reset()
    for e in range(num_epochs):
        for st in range(buffer.capacity):
            act, _ = agent.make_decision(
                params,
                jnp.expand_dims(pobs, axis=0),
            )
            # nobs, rew, term, trunc, _ = env.step(int(act))
            nobs, rew, term, trunc, _ = env.step(np.array(act))
            buffer.store(pobs, act, rew)
            ep_return += rew
            pobs = nobs
            if term or trunc:
                buffer.finish_episode()
                deposit_return.append(ep_return)
                average_return.append(sum(deposit_return) / len(deposit_return))
                print(f"episode: {ep+1}, steps: {st+1}, return: {ep_return}")
                ep += 1
                ep_return = 0
                pobs, _ = env.reset()
        buffer.finish_episode()
        replay = buffer.extract()
        # loss_val = loss_fn(params, rep.obs, rep.act, rep.ret)
        params, loss_val = agent.train_epoch(params, replay)
        print(f"\n---epoch {e+1} loss: {loss_val}---\n")
    env.close()
    plt.plot(average_return)
    plt.show()

    # VALIDATION
    # env = gym.make('CartPole-v1', render_mode='human')
    # env = gym.make('Pendulum-v1', g=9.81, render_mode='human')
    env = gym.make('LunarLander-v2', continuous=True, render_mode='human')
    pobs, _ = env.reset()
    term, trunc = False, False
    for _ in range(1000):
        act, qvals = agent.make_decision(
            params,
            jnp.expand_dims(pobs, axis=0),
        )
        # nobs, rew, term, trunc, _ = env.step(int(act))
        nobs, rew, term, trunc, _ = env.step(np.array(act))
        ep_return += rew
        pobs = nobs
        if term or trunc:
            print(f"\n---return: {ep_return}---\n")
            break
    env.close()

