import numpy as np
import jax.numpy as jnp
from flax import nnx
from tensorflow_probability.substrates.jax.distributions import Categorical, Normal


class GaussianNet(nnx.Module):
    """MLP Gaussian actor"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(8, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        mu = self.linear2(x)
        log_sigma = self.linear2(x)  # log(pi(a|s))
        pi = Normal(loc=mu, scale=jnp.exp(log_sigma))
        return pi


class ValueNet(nnx.Module):
    """MLP critic"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(8, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        v = self.linear2(x)
        return v


@nnx.jit
def resolve_and_assess(
    rngs: nnx.Rngs, actor: GaussianNet, critic: ValueNet, obs: np.ndarray
):
    pi = actor(jnp.expand_dims(obs, axis=0))
    act = pi.sample(seed=rngs)
    val = critic(obs)

    return act.squeeze(axis=0), val.squeeze()


rngs = nnx.Rngs(25)
actor = GaussianNet(rngs=rngs)
critic = ValueNet(rngs=rngs)
dobs = np.random.normal(loc=0.0, scale=1.0, size=(8,))
dact, dval = resolve_and_assess(rngs, actor, critic, dobs)
print(f"dummy action: {dact}, dummy value: {dval}")
