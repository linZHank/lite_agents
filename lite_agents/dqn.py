import random
import numpy as np
import tensorflow as tf
from collections import deque
import logging


################################################################
"""
Set GPU and memory
"""
# restrict GPU and memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
# set log level
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
################################################################


################################################################
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
    """

    def __init__(self, dim_obs: int, capacity: int = int(1e4)):
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
            pobs=tf.convert_to_tensor(self.pobs_buf[slices], dtype=tf.float32),
            acts=tf.convert_to_tensor(self.acts_buf[slices], dtype=tf.int32),
            rews=tf.convert_to_tensor(self.rews_buf[slices], dtype=tf.float32),
            disc=tf.convert_to_tensor(discount_rate * (1 - self.term_buf[slices]), dtype=tf.float32),
            nobs=tf.convert_to_tensor(self.nobs_buf[slices], dtype=tf.float32)
        )

        return data
################################################################


class Critic(tf.keras.Model):
    """Critic module evaluates value of states.
    Source: https://www.tensorflow.org/guide/keras/custom_layers_and_models
    TODO:
        - Add Args and Returns.
        - Speicify args and returns type
    """
    def __init__(self, dim_outputs, hidden_sizes=[128, 128], activation="relu", out_activation=None):
        super(Critic, self).__init__()
        # hyperparams
        self._dim_outputs = dim_outputs
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._out_activation = out_activation
        # construct critic network
        self.critic_net = tf.keras.Sequential()
        for size in hidden_sizes:
            self.critic_net.add(tf.keras.layers.Dense(size, activation=activation))
        self.critic_net.add(tf.keras.layers.Dense(dim_outputs, activation=out_activation))

    @tf.function
    def call(self, obs):
        value = tf.squeeze(self.critic_net(obs))

        return value


def polynomial_schedule(
    init_value: float,
    end_value: float,
    power: int,
    transition_steps: int,
    transition_begin: int = 0
):
    """Constructs a schedule with polynomial transition from init to end value.
    Source: https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py
    tODO: Move to utils.py

    Args:
        init_value: initial value for the scalar to be annealed.
        end_value: end value of the scalar to be annealed.
        power: the power of the polynomial used to transition from init to end.
        transition_steps: number of steps over which annealing takes place,
            the scalar starts changing at `transition_begin` steps and completes
            the transition by `transition_begin + transition_steps` steps.
            If `transition_steps <= 0`, then the entire annealing process is disabled
            and the value is held fixed at `init_value`.
        transition_begin: must be positive. After how many steps to start annealing
            (before this many steps the scalar value is held fixed at `init_value`).
    Returns:
        schedule: A function that maps step counts to values.
    """
    if transition_steps <= 0:
        logging.info(
            'A polynomial schedule was set with a non-positive `transition_steps` '
            'value; this results in a constant schedule with value `init_value`.')
        return lambda count: init_value

    if transition_begin < 0:
        logging.info(
            'An exponential schedule was set with a negative `transition_begin` '
            'value; this will result in `transition_begin` falling back to `0`.')
        transition_begin = 0

    def schedule(count):
        count = np.clip(count - transition_begin, 0, transition_steps)
        frac = 1 - count / transition_steps
        return (init_value - end_value) * (frac**power) + end_value

    return schedule


class DQNAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        learning_rate=3e-4,
        polyak=0.995,
        update_freq=100,
    ):
        # env related
        self._env = env
        self._dim_obs = env.observation_space.shape[0]  # not necessary
        self._num_acts = env.action_space.n
        # hyperparams
        self.epsilon_decay_schedule = polynomial_schedule(
            init_value=1.0,
            end_value=0.1,
            power=1,
            transition_steps=400,
        )
        # params
        self.gamma = gamma  # discount rate
        self.polyak = polyak
        self.periodic_update_freq = update_freq
        # init critic network
        self.critic = Critic(dim_outputs=self._num_acts)
        self.critic(np.expand_dims(env.observation_space.sample(), axis=0))
        self.online_params = self.critic.get_weights()
        self.target_params = self.online_params.copy()
        # self.targ_q = Critic(dim_obs, num_act, activation)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        # variable
        self.epsilon = 0.
        self.update_counter = 0
#
#     def linear_epsilon_decay(self, episode, decay_period, warmup_episodes):
#         episodes_left = decay_period + warmup_episodes - episode
#         bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
#         bonus = np.clip(bonus, 0.0, self.init_eps - self.final_eps)
#         self.epsilon = self.final_eps + bonus
#

    @tf.function
    def make_decision(self, obs, episode_count, eval_flag):
        obs = tf.expand_dims(obs, 0)  # add dummy batch
        q_vals = self.critic(obs)
        self._epsilon = self.epsilon_decay_schedule(episode_count)
        action = tf.argmax(q_vals, axis=-1, output_type=tf.dtypes.int32)
        if not eval_flag:
            if tf.random.uniform(shape=(), maxval=1) < self._epsilon:
                action = tf.random.uniform(
                    shape=[],
                    maxval=self._num_acts,
                    dtype=tf.dtypes.int32,
                )

        return action, q_vals

    # def update_params(self, data):
    #     # update critic
    #     with tf.GradientTape() as tape:
    #         tape.watch(self.q.trainable_weights)
    #         pred_qval = tf.math.reduce_sum(
    #             self.q(data["obs"]) * tf.one_hot(data["act"], self.num_act), axis=-1
    #         )
    #         targ_qval = data["rew"] + self.gamma * (
    #             1 - data["done"]
    #         ) * tf.math.reduce_sum(
    #             self.targ_q(data["nobs"])
    #             * tf.one_hot(
    #                 tf.math.argmax(self.q(data["nobs"]), axis=1), self.num_act
    #             ),
    #             axis=1,
    #         )  # double DQN trick
    #         loss_q = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval)
    #     logging.debug("q loss: {}".format(loss_q))
    #     grads = tape.gradient(loss_q, self.q.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.q.trainable_weights))
    #     self.update_counter += 1
    #     if self.polyak > 0:
    #         # Polyak average update target Q-nets
    #         q_weights_update = []
    #         for w_q, w_targ_q in zip(self.q.get_weights(), self.targ_q.get_weights()):
    #             w_q_upd = self.polyak * w_targ_q
    #             w_q_upd = w_q_upd + (1 - self.polyak) * w_q
    #             q_weights_update.append(w_q_upd)
    #         self.targ_q.set_weights(q_weights_update)
    #     else:
    #         if not self.update_counter % self.update_freq:
    #             self.targ_q.q_net.set_weights(self.q.q_net.get_weights())
    #             print(
    #                 "\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nTarget Q-net updated\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
    #             )
    #
    #     return loss_q

    # @tf.function
    def update_params(self, batch):
        # update target params
        if self.polyak > 0:
            for i in range(len(self.online_params)):
                self.target_params[i] = (1.0 - self.polyak) * self.online_params[i] \
                    + self.polyak * self.target_params[i]
        else:
            if not self.update_counter % self.update_freq:
                self.target_params = self.online_params.copy()
                print(
                    f"\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nTarget params updated at step: {self.update_counter}\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
                )
        # update online_params
        with tf.GradientTape() as tape:
            tape.watch(self.critic.trainable_weights)
            prev_qvals = self.critic(batch["pobs"])
            prev_qsa = prev_qvals * tf.one_hot(batch["acts"], self._num_acts)
            with tape.stop_recording():
                self.critic.set_weights(self.target_params)
                next_qvals = self.critic(batch["nobs"])
                self.critic.set_weights(self.online_params)
                duel_next_qvals = self.critic(batch["nobs"])
                next_acts = tf.one_hot(tf.argmax(duel_next_qvals, axis=-1), self._num_acts)
                next_qsa = next_qvals * next_acts
                target_q = batch["rews"] + batch["disc"] * tf.reduce_sum(next_qsa, axis=-1)
            pred_q = tf.reduce_sum(prev_qsa, axis=-1)
            td_error = tf.keras.losses.MSE(y_true=target_q, y_pred=pred_q)
        grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        self.online_params = self.critic.get_weights()
        self.update_counter += 1

        return td_error


# Uncomment following to test
import gymnasium as gym
from time import time

env = gym.make("LunarLander-v2")  # , render_mode="human")
# env = gym.make("CartPole-v0")
agent = DQNAgent(env)
buf = ReplayBuffer(dim_obs=env.observation_space.shape[0], capacity=int(1e6))
step_count = 0
episode_count = 0
t0 = time()
pobs, info = env.reset()
# print(f"initial observation: {pobs}, info: {info}")  # debug
term, trun = False, False
rew, episodic_return = 0, 0
deposit_return, averaged_return = [], []
for _ in range(100000):
    act, _ = agent.make_decision(
        obs=pobs, 
        episode_count=episode_count,
        eval_flag=False
    )
    nobs, rew, term, trun, info = env.step(act.numpy())
    # print(f"pobs: {pobs}\n act: {act}\n rew: {rew}\n term: {term}\n trun: {trun}\n nobs: {nobs}\n")  # debug
    buf.store(pobs, act, rew, term, nobs)
    episodic_return += rew
    pobs = nobs
    step_count += 1
    if buf.size > 1024:
        loss_value = agent.update_params(
            buf.sample(batch_size=1024, discount_rate=0.99),
        )
        # print(f"loss: {loss_value}")
    if term or trun:
        deposit_return.append(episodic_return)
        averaged_return.append(np.average(deposit_return))
        print(f"episode: {episode_count+1}, step: {step_count}, epsilon: {agent._epsilon} \nepisode return: {episodic_return} \nterminated: {term}, truncated: {trun}")
        print(f"average_return: {averaged_return[-1]} \n----\n")
        episode_count += 1
        pobs, _ = env.reset()
        term, trun = False, False
        rew, episodic_return = 0, 0
