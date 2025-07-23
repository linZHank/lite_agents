import jax.numpy as jnp
from scipy.signal import lfilter

r_list = [1.0] * 9
v_list = list(reversed(range(1, 10)))
nv_list = v_list[1:]
nv_list.append(0)
r_arr = jnp.array(r_list, dtype=jnp.float32)
v_arr = jnp.array(v_list, dtype=jnp.float32)
nv_arr = jnp.array(nv_list, dtype=jnp.float32)
print(r_arr)
print(v_arr)
print(nv_arr)
gamma = 0.9
advs = r_arr + gamma * nv_arr - v_arr
print(advs)
rev_advs = jnp.flip(advs, axis=0)
# print(rev_advs)
lam = 9.0  # try between 0.0 and 1.0
gae_advs = jnp.flip(lfilter([1], [1, -gamma * lam], rev_advs), axis=0)
print(gae_advs)
