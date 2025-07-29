from pathlib import Path
from flax import nnx
import orbax.checkpoint as ocp


class MLPNet(nnx.Module):
    """MLP critic"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(4, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        y = self.linear2(x)
        return y


model = MLPNet(rngs=nnx.Rngs(25))
state = nnx.split(model)
print("Model state: ")
nnx.display(model)

ckpt_path = Path(__file__).parent / "dummy_checkpoints" / "state"
ckpt_path.parent.mkdir(parents=True, exist_ok=True)
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(ckpt_path, state)
print(f"Model state saved at: {ckpt_path}")
