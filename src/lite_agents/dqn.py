from collections import namedtuple
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import copy
import flax.linen as nn
from flax.training import train_state
import optax
from distrax import Greedy, EpsilonGreedy


Batch = namedtuple('Batch', ['pobs', 'act', 'nobs', 'rew', 'gamma'])
Params = namedtuple('Params', ['online', 'stable'])


class ReplayBuffer(object):
    """A simple off-policy replay buffer."""

    def __init__(self, capacity, obs_shape):
        # Properties
        self.capacity = capacity
        # Variables
        self.loc = 0  # replay instance index
        self.occupied_size = 0
        # Replay storages
        self.buf_pobs = np.zeros(shape=[capacity]+list(obs_shape), dtype=np.float32)
        self.buf_acts = np.zeros(shape=(capacity, 1), dtype=int)
        self.buf_rews = np.zeros(shape=(capacity, 1), dtype=np.float32)
        self.buf_nobs = np.zeros_like(self.buf_pobs)
        self.buf_terms = np.zeros(shape=(capacity, 1), dtype=np.float32)

    def store(self, prev_obs, action, next_obs, reward, term_flag):
        self.buf_pobs[self.loc] = prev_obs
        self.buf_acts[self.loc] = action
        self.buf_nobs[self.loc] = next_obs
        self.buf_rews[self.loc] = reward
        self.buf_terms[self.loc] = term_flag
        self.loc = (self.loc + 1) % self.capacity
        self.occupied_size = min(self.occupied_size + 1, self.capacity)

    def sample(self, batch_size, discount=0.9):
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
        # sampled_terms = self.buf_terms[ids]
        sampled_discs = (1 - self.buf_terms[ids]) * discount
        sampled_batch = Batch(
            sampled_pobs,
            sampled_acts,
            sampled_nobs,
            sampled_rews,
            sampled_discs,
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


class DQNAgent:
    """RL agent powered by Deep-Q Network

    """
    def __init__(
        self,
        seed: int,
        obs_shape: tuple,
        num_actions: int,
        hidden_sizes: tuple = (64, 64),
        epsilon_decay_episodes: int = 100,
        lr: float = 1e-4,
        warmup_episodes: int = 10,
        polyak_step_size: float = 0.005,
    ):
        # Properties
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.lr = lr
        self.warmup_episodes = warmup_episodes
        self.polyak_step_size = polyak_step_size
        # Variables
        self.key = jax.random.PRNGKey(seed)
        self.ep_count = 0
        # Hyperparams scheduler
        self.epsilon_schedule = optax.linear_schedule(
            init_value=1.0,
            end_value=0.01,
            transition_steps=epsilon_decay_episodes,
            transition_begin=warmup_episodes,
        )
        self.epsilon = 1.0  # init_value 
        # self.lr_schedule = optax.linear_schedule(
        #     init_value=3e-4,
        #     end_value=1e-4,
        #     transition_steps=10000,
        # )
        # Q-Net and optimizer
        self.qnet = MLP(num_actions, hidden_sizes)
        self.params_online = self.qnet.init(
            self.key,
            jnp.expand_dims(jnp.ones(obs_shape), axis=0)
        )
        self.params_stable = copy(self.params_online, {})
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params_online)
        # Jitted methods
        self.loss_fn = jax.jit(self.loss_fn)
        self.polyak_update_fn = jax.jit(self.polyak_update_fn)
        # self.online_update_fn = jax.jit(self.online_update_fn)

    def loss_fn(self, params_online, replay_batch):
        @jax.vmap
        def double_q_error(act, rew, gamma, q_pred, q_next, q_duel):
            q_target = jax.lax.stop_gradient(
                rew + gamma * q_next[q_duel.argmax(axis=-1)]
            )
            td_error = q_target - q_pred[act]
            return td_error
        qval_pred = self.qnet.apply(params_online, replay_batch.pobs)
        qval_next = self.qnet.apply(self.params_stable, replay_batch.nobs)
        qval_duel = self.qnet.apply(params_online, replay_batch.nobs)
        qerr = double_q_error(
            replay_batch.act,
            replay_batch.rew,
            replay_batch.gamma,
            qval_pred,
            qval_next,
            qval_duel,
        )
        loss_value = optax.l2_loss(qerr).mean()
        return loss_value

    def online_update_fn(self, params_online, replay_batch):
        loss_grad_fn = jax.value_and_grad(self.loss_fn)
        loss_val, grads = loss_grad_fn(params_online, replay_batch)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params_online = optax.apply_updates(params_online, updates)
        return loss_val

    def polyak_update_fn(self):
        self.params_stable = optax.incremental_update(
            new_tensors=self.params_online,
            old_tensors=self.params_stable,
            step_size=0.01
        )

    def train_step(self, replay_batch):
        loss_value = self.online_update_fn(self.params_online, replay_batch)
        self.polyak_update_fn()
        return loss_value

    def make_decision(self, obs, eval_flag=True):
        qvalues = self.qnet.apply(
            self.params_online,
            obs,
        ).squeeze(axis=0)
        self.key, subkey = jax.random.split(self.key)
        if eval_flag:
            self.epsilon = 0.
        else:
            self.epsilon = self.epsilon_schedule(self.ep_count)
        act_greedy = Greedy(preferences=qvalues).sample(seed=subkey)
        act_sample = EpsilonGreedy(
            preferences=qvalues,
            epsilon=self.epsilon).sample(seed=subkey)
        action = jax.lax.select(
            pred=eval_flag,
            on_true=act_greedy,
            on_false=act_sample,
        )

        return action, qvalues


if __name__ == '__main__':
    import gymnasium as gym
    import matplotlib.pyplot as plt
    # SETUP
    env = gym.make('CartPole-v1')
    buffer = ReplayBuffer(10000, env.observation_space.shape)
    agent = DQNAgent(
        seed=19,
        obs_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
    )
    print(
        agent.qnet.tabulate(
            agent.key,
            env.observation_space.sample(),
            compute_flops=True,
            compute_vjp_flops=True
        )
    )  # view QNet structure in a table
    ep_return = 0
    deposit_return, average_return = [], []
    pobs, _ = env.reset()
    for st in range(5000):
        act, qvals = agent.make_decision(
            jnp.expand_dims(pobs, axis=0),
            eval_flag=False,
        )
        nobs, rew, term, trunc, _ = env.step(int(act))
        buffer.store(pobs, act, nobs, rew, term)
        ep_return += rew
        # print(f"previous observation: {pobs}")
        # print(f"action: {act}")
        # print(f"q-value: {qvals}")
        # print(f"next observation: {nobs}")
        # print(f"reward: {rew}")
        # print(f"termination flag: {term}")
        # print(f"truncated flag: {trunc}")
        if agent.ep_count >= agent.warmup_episodes:
            replay = buffer.sample(256)
            # loss_val = agent.loss_fn(agent.params_online, replay)
            # print(f"loss: {loss_val}")
            loss_val = agent.train_step(replay)
    #     est += 1
        pobs = nobs
        if term or trunc:
            agent.ep_count += 1
            deposit_return.append(ep_return)
            average_return.append(sum(deposit_return) / len(deposit_return))
            print(f"\n---episode: {agent.ep_count}, steps: {st+1}, epsilon:{agent.epsilon}, return: {ep_return}---\n")
            ep_return = 0
            pobs, _ = env.reset()
    #         print(f"\n---episode: {ep+1}, steps: {est}, return: {g}")
    #         print(f"total steps: {st+1}---\n")
    #         deposit_return.append(g)
    #         average_return.append(sum(deposit_return) / len(deposit_return))
    #         ep += 1
    #         est = 0
    #         g = 0
    #         o_0, i = env.reset()
    env.close()
    plt.plot(average_return)
    plt.show()
    #
    # # validation
    # env = gym.make('CartPole-v1', render_mode='human')
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
