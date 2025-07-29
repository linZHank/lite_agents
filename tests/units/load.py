from pathlib import Path
from flax import nnx
import orbax.checkpoint as ocp


class MLPNet(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        y = self.linear2(x)
        return y


# Construct abstract model
abstract_model = nnx.eval_shape(lambda: MLPNet(rngs=nnx.Rngs(25)))
graphdef, abstract_state = nnx.split(abstract_model)
print("The abstract NNX state (all leaves are abstract arrays):")
nnx.display(abstract_state)
# Restore state
ckpt_path = Path(__file__).parent / "dummy_checkpoints" / "state"
checkpointer = ocp.StandardCheckpointer()
state_restored = checkpointer.restore(ckpt_path, abstract_state)
print("NNX State restored: ")
nnx.display(state_restored)
# The model is now good to use!
model = nnx.merge(graphdef, state_restored)
