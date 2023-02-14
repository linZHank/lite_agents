import collections
import numpy as np
import jax
import jax.numpy as jnp
from jaxlib import xla_extension
import haiku as hk
import optax
import rlax
import matplotlib.pyplot as plt


################################################################
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
    """

    def __init__(self, dim_obs: int, capacity: int = int(1e5)):
        self.pobs_buf = np.zeros(shape=(capacity, dim_obs), dtype=np.float32)
        self.acts_buf = np.zeros(shape=capacity, dtype=np.int32)
        self.rews_buf = np.zeros(shape=capacity, dtype=np.float32)
        self.term_buf = np.zeros(shape=capacity, dtype=bool)
        self.nobs_buf = np.zeros(shape=(capacity, dim_obs), dtype=np.float32)
        self.ptr, self.size, self.capacity = 0, 0, capacity

    def store(self, pobs, act, rew, term, nobs):
        self.pobs_buf[self.ptr] = pobs
        self.nobs_buf[self.ptr] = nobs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.term_buf[self.ptr] = term
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size: int = 128, discount_rate: float = 0.99) -> dict:
        slices = np.random.randint(low=0, high=self.size, size=batch_size)
        data = dict(
            pobs=self.pobs_buf[slices],
            acts=self.acts_buf[slices],
            rews=self.rews_buf[slices],
            disc=np.multiply(discount_rate, (1 - self.term_buf[slices]), dtype=np.float32),
            nobs=self.nobs_buf[slices], 
        )

        return data
    
    def is_ready(self, batch_size):  # warm up trick
        return batch_size <= self.size
################################################################


# Declare DQN agent and trainable parameters
def transformed_mlp(output_size: int, hidden_sizes: list = [128, 128]) -> hk.Transformed:
    """Factory for a simple MLP network (for approximating Q-values)."""

    def forward(inputs: int):
        mlp = hk.nets.MLP(hidden_sizes + [output_size])

        return mlp(inputs)

    return hk.without_apply_rng(hk.transform(forward))

Params = collections.namedtuple("Params", "online, target")

class DQNAgent:
    """A simple DQN agent. Compatible with gym"""

    def __init__(
        self,
        env,
        update_freq,
        polyak,
        learning_rate,
    ):
        # env related
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # hyperparams
        self.epsilon_by_frame = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.01,
            power=1,
            transition_steps=500,
        )
        self.polyak = polyak
        self.update_freq = update_freq
        self.learning_rate = learning_rate
        # Neural net and optimiser.
        self.critic_net = transformed_mlp(output_size=int(self.action_space.n))
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

    def update_params(self, params, batch):
        """Periodic update online params.

        TODO: add polyak update
        """
        # target_params = optax.periodic_update(
        #     params.online,
        #     params.target,
        #     self.update_count,
        #     self.update_freq,
        # )
        target_params = optax.incremental_update(
            new_tensors=params.online,
            old_tensors=params.target,
            step_size=1 - self.polyak,
        )
        loss_value, loss_grads = jax.value_and_grad(self.loss_fn)(
            params.online, target_params, batch
        )  # but seems jax.grad only compute grads for first explicit arg
        updates, self.opt_state = self.optimizer.update(loss_grads, self.opt_state)
        online_params = optax.apply_updates(params.online, updates)
        self.update_count += 1

        return loss_value, Params(online_params, target_params)

    def loss_fn(
        self,
        online_params,
        target_params,
        data,
    ):
        prev_qval = self.critic_net.apply(online_params, data["pobs"])
        next_qval = self.critic_net.apply(target_params, data["nobs"])
        deul_qval = self.critic_net.apply(online_params, data["nobs"])
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(
            prev_qval, data["acts"], data["rews"], data["disc"], next_qval, deul_qval
        )
        return optax.l2_loss(td_error).mean()


# Uncomment following to test
import gymnasium as gym
from time import time

key_iter = hk.PRNGSequence(jax.random.PRNGKey(20))
env = gym.make("LunarLander-v2")  # , render_mode="human")
agent = DQNAgent(
    env=env,
    polyak=0.995,
    update_freq=200,
    learning_rate=1e-4,
)
params = agent.init_params(next(key_iter))
agent.init_optimizer(params)
buf = ReplayBuffer(dim_obs=env.observation_space.shape[0], capacity=int(1e6))
episode_count = 0
pobs, info = env.reset(seed=20)
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
            buf.sample(batch_size=1024, discount_rate=0.99),
        )
        # print(f"loss = {loss_value}")
    if term or trun:  # reset env if terminated or truncated
        deposit_return.append(episodic_return)
        averaged_return.append(np.average(deposit_return))
        print(f"episode: {episode_count+1}, step: {step_count+1}, epsilon: {epsilon} \nepisode return: {episodic_return} \nterminated: {term}, truncated: {trun}")
        print(f"averaged_return: {averaged_return[-1]}\n----\n")
        episode_count += 1
        pobs, _ = env.reset(seed=20)
        term, trun = False, False
        rew, episodic_return = 0, 0
t1 = time()
print(f"time consuming: {t1 - t0}")
plt.plot(averaged_return)
plt.show()
