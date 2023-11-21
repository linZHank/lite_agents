import jax
import jax.numpy as jnp
import numpy as np
from timeit import timeit


def set_np_array():
    a = np.zeros(shape=10, dtype=np.float32)
    for i in range(a.shape[0]):
        a[i] = np.random.normal()

    return a


def set_jnp_array():
    key = jax.random.PRNGKey(42)
    ja = jnp.zeros(shape=10, dtype=jnp.float32)
    for i in range(ja.shape[0]):
        key, subkey = jax.random.split(key)  # generate a new key, or sampled action won't change
        ja = ja.at[i].set(jax.random.normal(subkey))

    return ja


print(timeit(lambda: set_np_array(), number=1000))
print(timeit(lambda: set_jnp_array(), number=1000))

