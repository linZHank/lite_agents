from flax import nnx
import orbax.checkpoint as ocp


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


abstract_model = nnx.eval_shape(lambda: ValueNet(rngs=nnx.Rngs(25)))
graphdef, abstract_state = nnx.split(abstract_model)
print("The abstract NNX state (all leaves are abstract arrays):")
nnx.display(abstract_state)

checkpointer = ocp.StandardCheckpointer()
state_restored = checkpointer.restore(
    "/tmp/spinupax/2025-07-28-19-01/ac/checkpoints/critic/state_128", abstract_state
)
print("NNX State restored: ")
nnx.display(state_restored)

# The model is now good to use!
model = nnx.merge(graphdef, state_restored)
