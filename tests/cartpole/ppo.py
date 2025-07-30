from collections import namedtuple
import gymnasium as gym
import numpy as np
from scipy.signal import lfilter

import jax.numpy as jnp
from flax import nnx
import optax
from tensorflow_probability.substrates.jax.distributions import Categorical, Normal

import matplotlib.pyplot as plt

ReplayBuffer = namedtuple(
    "ReplayBuffer",
    "observations actions rewards values step_returns advantages, log_probas",
)
ExperienceBatch = namedtuple("ExperienceBatch", "obs act ret adv logp")


class PPOBuffer(ReplayBuffer):
    def store_step(self, obs, act, rew, val, logp):
        self.observations.append(obs)  # observation
        self.actions.append(act)  # action
        self.rewards.append(rew)  # reward
        self.values.append(val)  # value
        self.log_probas.append(logp)  # log probability of action

    def wrapup_episode(self, eoe_value, episode_len, discount=0.99, tradeoff=0.97):
        """
        Process raw data after an episode, calculate returns-to-go and GAE advantage estimations.
        Args:
            eoe_value: end of episode value estimation.
            episode_len: number of steps in the just-finished episode.
            discount (gamma): price of a reward in future will be penalized at present.
            tradeoff (lambda): balance variance and bias of advantage estimation.
                GAE(lambda=0): r_t + gamma V(s_{t+1}) - V(s_t), high bias low variance
                GAE(lambda=1): sum_{l=0}^{infty} gamma^l r+{t+1} - V(s_t), low bias high variance
        """
        next_vals = self.values[-episode_len + 1 :]
        next_vals.append(eoe_value)
        nv_arr = jnp.array(next_vals)
        r_arr = jnp.array(self.rewards[-episode_len:])
        v_arr = jnp.array(self.values[-episode_len:])
        # GAE-Lambda advantage
        td_errs = r_arr + discount * nv_arr - v_arr  # r_t + gamma V(s_{t+1}) - V(s_t)
        gae_advs = jnp.flip(
            lfilter([1], [1, -discount * tradeoff], jnp.flip(td_errs)), axis=0
        )
        self.advantages.extend(gae_advs.tolist())
        # Discounted returns to-go
        rev_ep_rews = self.rewards[-episode_len:][::-1]  # reversed episodic rewards
        drtg = lfilter([1], [1, -discount], rev_ep_rews)[::-1]  # discounted return togo
        self.step_returns.extend(drtg.tolist())

    def extract_experience(self):
        observations_batch = jnp.array(self.observations)  # NOTE: won't work under 1D
        actions_batch = jnp.array(self.actions)
        returns_batch = jnp.expand_dims(jnp.array(self.step_returns), axis=-1)
        advantages_batch = jnp.expand_dims(jnp.array(self.advantages), axis=-1)
        log_probas_batch = jnp.array(self.log_probas)

        experience_batch = ExperienceBatch(
            observations_batch,
            actions_batch,
            returns_batch,
            advantages_batch,
            log_probas_batch,
        )

        return experience_batch


class CategoricalPolicyNet(nnx.Module):
    """MLP Actor"""

    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        y = self.linear3(x)
        log_prob = nnx.log_softmax(y)  # log(pi(a|s))
        pi = Categorical(logits=jnp.expand_dims(log_prob, axis=1))

        return pi


class ValueNet(nnx.Module):
    """MLP critic"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        v = self.linear3(x)
        return v


@nnx.jit
def objective_fn(
    actor: CategoricalPolicyNet, experience_batch: ExperienceBatch, clip_constant: float
):
    policies = actor(experience_batch.obs)
    logpas = policies.log_prob(experience_batch.act)
    policy_ratios = jnp.exp(logpas - experience_batch.logp)  # pi/old_pi
    clipped_advs = (
        jnp.clip(policy_ratios, min=1 - clip_constant, max=1 + clip_constant)
        * experience_batch.adv
    )
    objectives = jnp.minimum(policy_ratios * experience_batch.adv, clipped_advs)

    return -objectives.mean()  # A(s_t, a_t) log(pi(a_t|s_t))


@nnx.jit
def loss_fn(critic: ValueNet, experience_batch: ExperienceBatch):
    pred_vals = critic(experience_batch.obs)
    targ_vals = experience_batch.ret
    l2_loss = optax.l2_loss(pred_vals, targ_vals)

    return l2_loss.mean()


@nnx.jit
def update_actor_params(
    actor: nnx.Module,
    actor_optimizer: nnx.Optimizer,
    experience_batch: ExperienceBatch,
    clip_constant: float = 0.2,
):
    grad_fn = nnx.value_and_grad(objective_fn)
    objective, grads = grad_fn(actor, experience_batch, clip_constant)
    actor_optimizer.update(grads)  # In-place updates.

    return objective


@nnx.jit
def update_critic_params(
    critic: ValueNet, critic_optimizer: nnx.Optimizer, experience_batch: ExperienceBatch
):
    grad_fn = nnx.value_and_grad(loss_fn)
    v_loss, grads = grad_fn(critic, experience_batch)
    critic_optimizer.update(grads)  # In-place updates.

    return v_loss


@nnx.jit
def resolve_and_assess(
    rngs: nnx.Rngs,
    actor: CategoricalPolicyNet,
    critic: ValueNet,
    obs: np.ndarray,
):
    pi = actor(jnp.expand_dims(obs, axis=0))
    act = pi.sample(seed=rngs)
    log_prob_act = pi.log_prob(act)
    val = critic(obs)

    return act.squeeze(axis=0), val.squeeze(), log_prob_act.squeeze(axis=0)


# SETUP
env = gym.make("CartPole-v1", render_mode="rgb_array")
rngs = nnx.Rngs(25)
actor = CategoricalPolicyNet(rngs=rngs)
critic = ValueNet(rngs=rngs)
actor_optimizer = nnx.Optimizer(actor, optax.adamw(3e-4))
critic_optimizer = nnx.Optimizer(critic, optax.adamw(1e-4))
buffer = PPOBuffer([], [], [], [], [], [], [])
journal_learn = {
    "episode_idx": 0,
    "step_idx": 0,
    "episode_len": [0],
    "deposit_return": [0.0],
    "averaged_return": [],
}

# LOOP
last_obs, info = env.reset()
for e in range(32):
    for st in range(6 * env.spec.max_episode_steps):
        # Play a step
        pred_act, last_val, old_logp = resolve_and_assess(rngs, actor, critic, last_obs)
        act = pred_act.squeeze()
        next_obs, rew, term, trunc, info = env.step(np.array(act))
        buffer.store_step(last_obs, pred_act, rew, last_val, old_logp)
        last_obs = next_obs.copy()
        # Step statistics
        journal_learn["step_idx"] += 1  # TODO: update journal in a util function
        journal_learn["episode_len"][-1] += 1
        journal_learn["deposit_return"][-1] += rew
        # Wrap up episode
        if term or trunc:
            if trunc:
                eoe_val = jnp.squeeze(critic(last_obs))
            else:
                eoe_val = jnp.zeros(shape=())
            buffer.wrapup_episode(eoe_val, journal_learn["episode_len"][-1])
            # Episode statistics
            journal_learn["episode_idx"] += 1
            journal_learn["averaged_return"].append(
                sum(journal_learn["deposit_return"])
                / len(journal_learn["deposit_return"])
            )
            # TODO: need a logger
            print(
                f"---\nepisode: {journal_learn['episode_idx']}, length: {journal_learn['episode_len'][-1]}, return: {journal_learn['deposit_return'][-1]}\n---\n"
            )
            # Reset episode
            last_obs, info = env.reset()
            journal_learn["episode_len"].append(0)
            journal_learn["deposit_return"].append(0.0)
            if st > 5 * env.spec.max_episode_steps:
                break
    # Wrap up epoch
    print(
        f"===\nepoch {e + 1} \n\ttotal steps: {journal_learn['step_idx']}\n\taveraged return: {journal_learn['averaged_return'][-1]}\n==="
    )
    experience_batch = buffer.extract_experience()
    for _ in range(16):
        obj_inv = update_actor_params(actor, actor_optimizer, experience_batch)
        print(f"Policy objective: {-obj_inv}")  # TODO: Metrics
    for _ in range(80):
        v_loss = update_critic_params(critic, critic_optimizer, experience_batch)
        print(f"Value estimation loss: {v_loss}")  # TODO: Metrics
    buffer = PPOBuffer([], [], [], [], [], [], [])

# TODO: need a plotter
plt.plot(journal_learn["averaged_return"])
plt.grid(visible=True)
plt.show()
# plt.savefig(Path(__file__).parent.joinpath(f"{env_name}.png"))

# Evaluation
env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()
episode_return = 0.0
for _ in range(env.spec.max_episode_steps):
    pred_act, _, _ = resolve_and_assess(rngs, actor, critic, obs)
    act = pred_act.squeeze()
    obs, rew, term, trunc, _ = env.step(np.array(act))
    episode_return += rew
    if term or trunc:
        print(f"\n---return: {episode_return}---\n")
        break
