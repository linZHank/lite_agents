import gymnasium as gym
import jax
import jax.numpy as jnp
import flax.linen as nn
from distrax import Categorical
import matplotlib.pyplot as plt


class MLP(nn.Module):
    num_outputs: int
    hidden_sizes: tuple = (64, 64)

    @nn.compact
    def __call__(self, inputs):
        dtype = jnp.float32
        x = inputs.astype(dtype)
        for i, size in enumerate(self.hidden_sizes):
            z = nn.Dense(features=size, name='hidden'+str(i+1), dtype=dtype)(x)
            x = nn.relu(z)
        logits = nn.Dense(features=self.num_outputs, name='logits')(x)
        return logits

# SETUP
key = jax.random.PRNGKey(19)
env = gym.make('CartPole-v1')
policy_net = MLP(env.action_space.n)
params = policy_net.init(
    key,
    jnp.expand_dims(env.observation_space.sample(), axis=0)
)

def make_decision(key, params, obs):
    logits = policy_net.apply(params, obs).squeeze(axis=0)
    distribution = Categorical(logits=logits)
    act = distribution.sample(seed=key)
    logp_a = distribution.log_prob(act)
    return act, logp_a


# LOOP
ep, ep_return = 0, 0
deposit_return, average_return = [], []
pobs, _ = env.reset()
key, subkey = jax.random.split(key)
for st in range(500):
    key, subkey = jax.random.split(key)
    act, logp_a = make_decision(
        subkey,
        params,
        jnp.expand_dims(pobs, axis=0),
    )
    print(act, logp_a)
    # act = env.action_space.sample()
    nobs, rew, term, trunc, _ = env.step(int(act))
    ep_return += rew
    pobs = nobs
    if term or trunc:
        deposit_return.append(ep_return)
        average_return.append(sum(deposit_return) / len(deposit_return))
        print(f"\n---episode: {ep+1}, steps: {st+1}, return: {ep_return}---\n")
        ep += 1
        ep_return = 0
        pobs, _ = env.reset()
env.close()
plt.plot(average_return)
plt.show()

# validation
env = gym.make('CartPole-v1', render_mode='human')
pobs, _ = env.reset()
term, trunc = False, False
for _ in range(500):
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

