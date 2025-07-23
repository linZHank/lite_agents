import jax.numpy as jnp
from scipy.signal import lfilter

r = jnp.arange(1, 10)
print(r)
gamma = 0.9


def discount_cumsum(x, gamma):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


dcs_r = discount_cumsum(r, gamma)
rev_r = r[-9:][::-1]  # reversed stepwise returns
print(rev_r)
lf_r = lfilter([1], [1, -gamma], rev_r)[::-1]  # discounted return togo

print(dcs_r)
print(lf_r)
