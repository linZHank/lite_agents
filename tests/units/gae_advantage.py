import jax.numpy as jnp

v_list = list(range(1, 10))
nv_list = v_list[1:]
nv_list.append(0)
v = jnp.arange(1, 10, dtype=jnp.float32)
