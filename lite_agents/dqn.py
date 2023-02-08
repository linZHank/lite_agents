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
    """A simple off-policy replay buffer."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

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
        value = tf.squeeze(self.critic_net(obs), axis=0)

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
        update_type="polyak",
        polyak=0.995,
        update_freq=100,
    ):
        # env related
        self._env = env
        # self.observation_space = observation_space
        # self.action_space = action_space
        # hyperparams
        self.epsilon_decay_schedule = polynomial_schedule(
            init_value=1.0,
            end_value=0.01,
            power=1,
            transition_steps=500,
        )
        # params
        self.gamma = gamma  # discount rate
        self.polyak = polyak
        self.periodic_update_freq = update_freq
        self.init_eps = 1.0
        self.final_eps = 0.1
        # model
        self.critic_net = Critic(dim_outputs=env.action_space.n)
        self.online_params = None
        self.target_params = None
        # self.targ_q = Critic(dim_obs, num_act, activation)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        # variable
        self.epsilon = self.init_eps
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
        q_vals = self.critic_net(obs)
        self._epsilon = self.epsilon_decay_schedule(episode_count)
        action = tf.argmax(q_vals, axis=-1, output_type=tf.dtypes.int32)
        if not eval_flag:
            if tf.random.uniform(shape=(), maxval=1) < self._epsilon:
                action = tf.random.uniform(
                shape=[], maxval=self._env.action_space.n, dtype=tf.dtypes.int32
            )
        
        return action, q_vals
#
#     def train_one_batch(self, data):
#         # update critic
#         with tf.GradientTape() as tape:
#             tape.watch(self.q.trainable_weights)
#             pred_qval = tf.math.reduce_sum(
#                 self.q(data["obs"]) * tf.one_hot(data["act"], self.num_act), axis=-1
#             )
#             targ_qval = data["rew"] + self.gamma * (
#                 1 - data["done"]
#             ) * tf.math.reduce_sum(
#                 self.targ_q(data["nobs"])
#                 * tf.one_hot(
#                     tf.math.argmax(self.q(data["nobs"]), axis=1), self.num_act
#                 ),
#                 axis=1,
#             )  # double DQN trick
#             loss_q = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval)
#         logging.debug("q loss: {}".format(loss_q))
#         grads = tape.gradient(loss_q, self.q.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.q.trainable_weights))
#         self.update_counter += 1
#         if self.polyak > 0:
#             # Polyak average update target Q-nets
#             q_weights_update = []
#             for w_q, w_targ_q in zip(self.q.get_weights(), self.targ_q.get_weights()):
#                 w_q_upd = self.polyak * w_targ_q
#                 w_q_upd = w_q_upd + (1 - self.polyak) * w_q
#                 q_weights_update.append(w_q_upd)
#             self.targ_q.set_weights(q_weights_update)
#         else:
#             if not self.update_counter % self.update_freq:
#                 self.targ_q.q_net.set_weights(self.q.q_net.get_weights())
#                 print(
#                     "\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nTarget Q-net updated\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
#                 )
#
#         return loss_q


# Uncomment following to test
import gymnasium as gym
from time import time

env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("CartPole-v0")
agent = DQNAgent(env=env)
buf = ReplayBuffer(capacity=int(1e6))
step_count = 0
episode_count = 0
t0 = time()
pobs, info = env.reset()
# print(f"initial observation: {pobs}, info: {info}")  # debug
term, trun = False, False
rew, episodic_return = 0, 0
deposit_return, averaged_return = [], []
for _ in range(2000):
    act, _ = agent.make_decision(
        obs=pobs, 
        episode_count=episode_count,
        eval_flag=False
    )
    nobs, rew, term, trun, info = env.step(act.numpy())
    print(f"pobs: {nobs}\n act: {act}\n rew: {rew}\n term: {term}\n trun: {trun}\n nobs: {nobs}\n")  # debug
    episodic_return += rew
    pobs = nobs
    step_count += 1
    if term or trun:
        deposit_return.append(episodic_return)
        averaged_return.append(np.average(deposit_return))
        print(f"episode: {episode_count+1}, step: {step_count}, epsilon: {agent._epsilon} \nepisode return: {episodic_return} \nterminated: {term}, truncated: {trun}")
        episode_count += 1
        pobs, _ = env.reset()
        term, trun = False, False
        rew, episodic_return = 0, 0
