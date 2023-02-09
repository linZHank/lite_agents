"""A simple double-DQN agent."""

import collections
import random
import numpy as np

import jax
import jax.numpy as jnp
from jaxlib import xla_extension
import haiku as hk
import optax
import rlax

import matplotlib.pyplot as plt


def transformed_mlp(output_size: int, hidden_sizes: list = [128, 128]) -> hk.Transformed:
    """Factory for a simple MLP network (for approximating Q-values)."""

    def forward(inputs: int):
        mlp = hk.nets.MLP(hidden_sizes + [output_size])

        return mlp(inputs)

    return hk.without_apply_rng(hk.transform(forward))


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def store(self, prev_obs, action, reward, terminated, next_obs):
        if action is not None:
            self.buffer.append(
                (
                    prev_obs,
                    action,
                    reward,
                    terminated,
                    next_obs,
                )
            )

    def sample(self, batch_size, discount_factor):

        pobs, acts, rews, terms, nobs = zip(*random.sample(
            self.buffer,
            batch_size,
        ))

        return (
            np.stack(pobs, dtype=np.float32),
            np.asarray(acts, dtype=np.int32),
            np.asarray(rews, dtype=np.float32),
            (1 - np.asarray(terms, dtype=np.float32)) * discount_factor,
            np.stack(nobs, dtype=np.float32),
        )

    def is_ready(self, batch_size):  # warm up trick
        return batch_size <= len(self.buffer)


# Declare DQN agent and trainable parameters
Params = collections.namedtuple("Params", "online, target")
class DQNAgent:
    """A simple DQN agent. Compatible with gym"""

    def __init__(
        self,
        observation_space,
        action_space,
        update_period,
        learning_rate,
    ):
        # env related
        self.observation_space = observation_space
        self.action_space = action_space
        # hyperparams
        self.epsilon_by_frame = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.01,
            power=1,
            transition_steps=500,
        )
        self.update_period = update_period
        self.learning_rate = learning_rate
        # Neural net and optimiser.
        self.critic_net = transformed_mlp(output_size=int(action_space.n))
        self.optimizer = optax.adam(learning_rate)
        # variables
        self.update_count = 0
        self.epsilon = None
        self.online_params = None
        self.target_params = None
        self.opt_state = None
        # Jitting for speed.
        self.make_decision = jax.jit(self.make_decision)
        self.update_params = jax.jit(self.update_params)

    def init_params(self, key: xla_extension.DeviceArray):
        sample_input = self.observation_space.sample()
        sample_input = jnp.expand_dims(sample_input, 0)
        online_params = self.critic_net.init(key, sample_input)

        return Params(online_params, online_params)

    def init_optimizer(self, params):
        self.opt_state = self.optimizer.init(params.online)

    # @jax.jit
    def make_decision(self, key, params, obs, episode_count, eval_flag):
        """pi(a|s)
        TODO:
            add warm up
            rewrite epsilon greedy w/o rlax
        """
        obs = jnp.expand_dims(obs, 0)  # add dummy batch
        q_val = jnp.squeeze(self.critic_net.apply(params.online, obs))
        epsilon = self.epsilon_by_frame(episode_count)
        sampled_action = rlax.epsilon_greedy(epsilon).sample(key, q_val)
        greedy_action = rlax.greedy().sample(key, q_val)
        action = jax.lax.select(eval_flag, greedy_action, sampled_action)

        return action, q_val, epsilon

    def update_params(self, params, data):
        """Periodic update online params.

        TODO: add polyak update
        """
        target_params = optax.periodic_update(
            params.online,
            params.target,
            self.update_count,
            self.update_period,
        )
        loss_value, loss_grads = jax.value_and_grad(self.loss_fn)(
            params.online, target_params, *data
        )  # but seems jax.grad only compute grads for first explicit arg
        updates, self.opt_state = self.optimizer.update(loss_grads, self.opt_state)
        online_params = optax.apply_updates(params.online, updates)
        self.update_count += 1

        return loss_value, Params(online_params, target_params)

    def loss_fn(
        self,
        online_params,
        target_params,
        pobs_batch,
        acts_batch,
        rews_batch,
        disc_batch,
        nobs_batch,
    ):
        prev_qval = self.critic_net.apply(online_params, pobs_batch)
        next_qval = self.critic_net.apply(target_params, nobs_batch)
        deul_qval = self.critic_net.apply(online_params, nobs_batch)
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(
            prev_qval, acts_batch, rews_batch, disc_batch, next_qval, deul_qval
        )
        return optax.l2_loss(td_error).mean()


# Uncomment following to test
import gymnasium as gym
from time import time

key_iter = hk.PRNGSequence(jax.random.PRNGKey(20))
env = gym.make("LunarLander-v2")  # , render_mode="human")
# env = gym.make("CartPole-v0")
agent = DQNAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    update_period=50,
    learning_rate=3e-4,
)
params = agent.init_params(next(key_iter))
agent.init_optimizer(params)
buf = ReplayBuffer(capacity=int(1e6))
episode_count = 0
pobs, info = env.reset()
print(f"initial observation: {pobs}, info: {info}")  # debug
term, trun = False, False
rew, episodic_return = 0, 0
deposit_return, averaged_return = [], []
t0 = time()
for step_count in range(200000):
    action, q_val, epsilon = agent.make_decision(
        key=next(key_iter),
        params=params,
        obs=pobs,
        episode_count=episode_count,
        eval_flag=False,
    )
    act = int(action)
    nobs, rew, term, trun, info = env.step(act)
    # print(f"pobs: {nobs}\n act: {act}\n rew: {rew}\n term: {term}\n trun: {trun}\n nobs: {nobs}\n")  # debug
    buf.store(pobs, act, rew, term, nobs)
    episodic_return += rew
    pobs = nobs.copy()
    if buf.is_ready(batch_size=1024):
        loss_value, params = agent.update_params(
            params,
            buf.sample(batch_size=1024, discount_factor=0.99),
        )
        # print(f"loss = {loss_value}")
    if term or trun:  # reset env if terminated or truncated
        deposit_return.append(episodic_return)
        averaged_return.append(np.average(deposit_return))
        print(f"episode: {episode_count+1}, step: {step_count+1}, epsilon: {epsilon} \nepisode return: {episodic_return} \nterminated: {term}, truncated: {trun}")
        print(f"averaged_return: {averaged_return[-1]}\n----\n")
        episode_count += 1
        pobs, _ = env.reset()
        term, trun = False, False
        rew, episodic_return = 0, 0
t1 = time()
print(f"time consuming: {t1 - t0}")
