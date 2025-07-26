import numpy as np
import jax.numpy as jnp
from flax import nnx
from tensorflow_probability.substrates.jax.distributions import Categorical, Normal


class GaussianNet(nnx.Module):
    """MLP Gaussian distributions"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(8, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        mu = self.linear2(x)
        log_sigma = self.linear2(x)  # log(pi(a|s))
        pi = Normal(loc=mu, scale=jnp.exp(log_sigma))

        return pi


class CategoricalNet(nnx.Module):
    """MLP Categorical distributions"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(8, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 4, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        y = self.linear2(x)
        log_prob = nnx.log_softmax(y)  # log(pi(a|s))
        pi = Categorical(logits=jnp.expand_dims(log_prob, axis=1))

        return pi


rngs = nnx.Rngs(25)
dobs = np.random.uniform(-1.0, 1.0, size=(3, 8))
print(f"dummy obs batch: \n{dobs}")
dact = np.random.randint(0, 4, (3, 1))  # HACK:
# dact = np.random.normal(loc=0.0, scale=1.0, size=(3, 2))
print(f"dummy act batch: \n{dact}")
dadv = np.arange(3, dtype=np.float32).reshape(3, 1)  # HACK:
print(f"dummy advantage batch: \n{dadv}")


# Categorical
cdm = CategoricalNet(rngs=rngs)  # define prob model
# nnx.display(cdm)
cd = cdm(dobs)
print(f"distributions: {cd}")
logp = cd.log_prob(dact)
print(f"log probability of dummy act: \n{logp}")
j = logp * dadv
print(f"objective: {j}")

# Gaussian
# nd = GaussianNet(rngs=rngs)
# # nnx.display(cd)
# pi = nd(dobs)
# print(f"distribution: {pi}")
# lp = pi.log_prob(dact)
# print(f"log probability of dummy act: {lp}")
# j = lp * dadv
# print(f"objective: {j}")
