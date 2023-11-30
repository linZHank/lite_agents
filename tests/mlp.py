import jax
from lite_agents.dqn import MLP


key = jax.random.PRNGKey(19)
model = MLP(num_outputs=2, hidden_sizes=[4, 3])
dummy_inputs = jax.random.normal(key, shape=(5, 7))
print(dummy_inputs)
dummy_params = model.init(key, dummy_inputs)
print(dummy_params)
print(model.apply(dummy_params, dummy_inputs))
