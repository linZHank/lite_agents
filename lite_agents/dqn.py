""" 
A DQN agent class 
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
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


def mlp(num_outputs: int, hidden_sizes: list = [128, 128], activate_fn: str = "relu", output_activate_fn: str = None):
    model = keras.Sequential()
    # model.add(keras.Input(shape=(num_inputs, )))
    for layer_size in hidden_sizes:
        model.add(keras.layers.Dense(layer_size, activation=activate_fn))
    model.add(keras.layers.Dense(num_outputs, activation=output_activate_fn))

    return model

def polynomial_schedule(
    init_value: float,
    end_value: float,
    power: int,
    transition_steps: int,
    transition_begin: int = 0
):
  """Constructs a schedule with polynomial transition from init to end value.
  Source: https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py
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
        observation_space,
        action_space,
        activate_fn="relu",
        gamma=0.99,
        learning_rate=3e-4,
        update_type="polyak",
        polyak=0.995,
        update_freq=100,
    ):
        # env related
        self.observation_space = observation_space
        self.action_space = action_space
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
        self.critic_net = mlp(num_outputs=action_space.n)
        self.online_params = None
        self.target_params = None
        # self.targ_q = Critic(dim_obs, num_act, activation)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        # variable
        self.epsilon = self.init_eps
        self.update_counter = 0

    def linear_epsilon_decay(self, episode, decay_period, warmup_episodes):
        episodes_left = decay_period + warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0.0, self.init_eps - self.final_eps)
        self.epsilon = self.final_eps + bonus

    def make_decision(self, obs, episode_count, eval_flag):
        obs = tf.expand_dims(obs, 0)  # add dummy batch
        q_vals = tf.squeeze(self.critic_net(obs))
        epsilon = self.epsilon_decay_schedule(episode_count)
        if np.random.rand() > self.epsilon:
            a = tf.argmax(self.q(obs), axis=-1)
        else:
            a = tf.random.uniform(
                shape=[1, 1], maxval=self.num_act, dtype=tf.dtypes.int32
            )
        return a

    def train_one_batch(self, data):
        # update critic
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_weights)
            pred_qval = tf.math.reduce_sum(
                self.q(data["obs"]) * tf.one_hot(data["act"], self.num_act), axis=-1
            )
            targ_qval = data["rew"] + self.gamma * (
                1 - data["done"]
            ) * tf.math.reduce_sum(
                self.targ_q(data["nobs"])
                * tf.one_hot(
                    tf.math.argmax(self.q(data["nobs"]), axis=1), self.num_act
                ),
                axis=1,
            )  # double DQN trick
            loss_q = tf.keras.losses.MSE(y_true=targ_qval, y_pred=pred_qval)
        logging.debug("q loss: {}".format(loss_q))
        grads = tape.gradient(loss_q, self.q.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_weights))
        self.update_counter += 1
        if self.polyak > 0:
            # Polyak average update target Q-nets
            q_weights_update = []
            for w_q, w_targ_q in zip(self.q.get_weights(), self.targ_q.get_weights()):
                w_q_upd = self.polyak * w_targ_q
                w_q_upd = w_q_upd + (1 - self.polyak) * w_q
                q_weights_update.append(w_q_upd)
            self.targ_q.set_weights(q_weights_update)
        else:
            if not self.update_counter % self.update_freq:
                self.targ_q.q_net.set_weights(self.q.q_net.get_weights())
                print(
                    "\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nTarget Q-net updated\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
                )

        return loss_q
