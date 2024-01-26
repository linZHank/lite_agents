from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from distrax import Categorical
from scipy.signal import lfilter


Replay = namedtuple('Replay', ['obs', 'act', 'ret'])

class ReplayBuffer(object):
    """A simple on-policy replay buffer."""

    def __init__(self, capacity: int):
        # Variables
        self.loc = 0  # buffer instance index
        self.ep_init_loc = 0  # episode initial index
        # Properties
        self.capacity = capacity
        self.shape_obs = env.observation_space.shape
        self.shape_act = env.action_space.shape
        if not len(self.shape_act):
            self.num_act = env.action_space.n
        else:
            self.num_act = None
        # Replay storages
        self.buf_obs = np.zeros(shape=[capacity]+list(shape_obs), dtype=np.float32)
        self.buf_acts = np.zeros(shape=(capacity, 1), dtype=int)
        self.buf_rews = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_rets = np.zeros_like(self.buf_rews)

    def store(self, observation, action, reward):
        assert self.loc < self.capacity
        self.buf_obs[self.loc] = observation
        self.buf_acts[self.loc] = action
        self.buf_rews[self.loc] = reward
        self.loc += 1

    def finish_episode(self, discount=0.98):
        """ End of episode process
        Call this at the end of a trajectory, to compute the return-to-go.
        """
        def compute_rtgs(rews):
            return lfilter([1], [1, -discount], rews[::-1], axis=0,)[::-1]

        ep_slice = slice(self.ep_init_loc, self.loc)
        self.buf_rets[ep_slice] = compute_rtgs(self.buf_rews[ep_slice])
        self.ep_init_loc = self.loc

    def dump(self):
        """Get on-policy replay experience
        """
        replay = Replay(
            self.buf_obs,
            self.buf_acts,
            self.buf_rets,
        )
        # clean up replay buffer for next epoch
        self.__init__(self.capacity, self.obs_shape, self.act_shape, self.num_act)
        return replay

class CategoricalPolicyNet(nn.Module):
    """An MLP for discrete action space
    """
    num_acts: int
    hidden_sizes: tuple

    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32)
        for i, size in enumerate(self.hidden_sizes):
            z = nn.Dense(features=size, name='hidden'+str(i+1), dtype=dtype)(x)
            x = nn.relu(z)
        logits = nn.Dense(features=self.num_acts, name='logits')(x)
        return logits

class GaussianPolicyNet(nn.Module):
    """An MLP for continuous action space
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

class VPGAgent:
    """RL agent powered by REINFORCE

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
        self.num_act = None
        if not len(self.act_shape):  # discrete
            self.num_act = env.action_space.n
        self.lr = learning_rate
        # Policy network
        if self.num_act:
            self.actor = CategoricalPolicyNet(self.num_act, hidden_sizes)
        else:
            self.actor = GaussianPolicyNet(self.obs_shape[0], hidden_sizes)

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
        logits = self.actor.apply(params, obs).squeeze(axis=0)
        distribution = Categorical(logits=logits)
        act = distribution.sample(seed=subkey)
        logp_a = distribution.log_prob(act)
        return act, logp_a

    def loss_fn(self, params, replay):
        logits = self.policy_net.apply(params, replay.obs)
        distributions = Categorical(logits=logits)
        logpas = distributions.log_prob(replay.act.squeeze())  # squeeze actions data
        return -(logpas * replay.ret.squeeze()).mean()  # squeeze returns data

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
    env = gym.make('CartPole-v1')
    buffer = ReplayBuffer(
        capacity=500,
        obs_shape=env.observation_space.shape,
        act_shape=env.action_space.shape,
        num_act=env.action_space.n,
    )
    agent = VPGAgent(
        seed=19,
        env=env,
    )
    params = agent.init_params()
    agent.init_optimizer(params)

    num_epochs = 100
    ep, ep_return = 0, 0
    deposit_return, average_return = [], []
    pobs, _ = env.reset()
    for e in range(num_epochs):
        for st in range(buffer.capacity):
            act, logp = agent.make_decision(
                params,
                jnp.expand_dims(pobs, axis=0),
            )
            # print(act, logp)
            nobs, rew, term, trunc, _ = env.step(int(act))
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
        replay = buffer.dump()
        # loss_val = loss_fn(params, rep.obs, rep.act, rep.ret)
        params, loss_val = agent.train_epoch(params, replay)
        print(f"\n---epoch {e+1} loss: {loss_val}---\n")
    env.close()
    plt.plot(average_return)
    plt.show()

    # VALIDATION
    env = gym.make('CartPole-v1', render_mode='human')
    pobs, _ = env.reset()
    term, trunc = False, False
    for _ in range(500):
        act, qvals = agent.make_decision(
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

