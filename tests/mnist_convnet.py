import sys
from torchvision import datasets
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from flax import linen as nn
from flax.training import train_state 
import optax
import numpy as np


# Hyper-parameters
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
MOMENTUM = 0.9
SEED = 19

# Load data
# torch_transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ]
# )
train_dataset = datasets.MNIST(
    sys.path[0] + "/data",
    train=True,
    download=True,
    # transform=torch_transform,
)
test_dataset = datasets.MNIST(
    sys.path[0] + "/data",
    train=False,
    download=True,
    # transform=torch_transform,
)
# loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# loader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Check data
# print(train_dataset.data.shape)  # 64x1x28x28
# print(train_dataset.targets.shape)  # 64


class ConvNet(nn.Module):
    """A simple MLP model"""

    @nn.compact
    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)

        return x


model = ConvNet()
print(model.tabulate(
    jax.random.key(0),
    jnp.ones((1, 28, 28, 1)),
    compute_flops=True,
    compute_vjp_flops=True
))


def create_train_state(model, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']  # initialize parameters by passing a template image
    tx = optax.sgd(learning_rate, momentum)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    return state


@jax.jit
def loss_fn(params, batch):
    logits = state.apply_fn({'params': params}, batch['image'])
    loss_val = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()

    return loss_val


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    vals, grads = grad_fn(state.params, batch)
    state = state.apply_gradients(grads=grads)

    return state, vals


def train_epoch(dataset, state, batch_size, shuffle=True, print_every=100):
    if batch_size > dataset.__len__():
        batch_size = dataset.__len__()
    num_batches = int(np.ceil(dataset.__len__() / batch_size))
    ids = np.arange(dataset.__len__())
    if shuffle:
        np.random.shuffle(np.arange(dataset.__len__()))
    X = dataset.data.numpy().reshape(-1, 28, 28, 1)
    y = dataset.targets.numpy()
    batch = {}
    losses = []
    accuracies = []

    for step in range(num_batches):
        start_id = step * batch_size
        end_id = (step + 1) * batch_size
        if step == num_batches - 1:
            end_id = len(ids)
        batch['image'] = jnp.array(X[ids[start_id:end_id]])
        batch['label'] = jnp.array(y[ids[start_id:end_id]])
        state, loss = train_step(state, batch)
        losses.append(loss)
        logits = state.apply_fn({'params': state.params}, batch['image'])
        pred = logits.argmax(axis=-1)
        acc = jnp.sum(pred == batch['label']) / batch['label'].size
        accuracies.append(acc)
        if not (step + 1) % print_every:
            print(f"batch {step+1} train loss: {loss}, train accuracy:{acc}")

    return state, losses, accuracies


def test_epoch(state):
    batch = {}
    losses = []
    accuracies = []
    for _, (X, y) in enumerate(dataloader):
        batch['image'] = jnp.array(X)
        batch['label'] = jnp.array(y)
        loss = loss_fn(state.params, batch)
        losses.append(loss)
        logits = state.apply_fn({'params': state.params}, batch['image'])
        pred = logits.argmax(axis=-1)
        acc = jnp.sum(pred == batch['label']) / batch['label'].size
        accuracies.append(acc)

    return losses, accuracies


# SETUP
key = jax.random.PRNGKey(SEED)
state = create_train_state(model, key, LEARNING_RATE, MOMENTUM)
losses_train, losses_test = [], []
# print(f"steps / epoch: {num_steps_per_epoch}")


# LOOP
for i in range(NUM_EPOCHS):
    state, losses_train, accuracies_train = train_epoch(train_dataset, state, BATCH_SIZE)
    epoch_loss_train = sum(losses_train) / len(losses_train)
    epoch_acc_train = sum(accuracies_train) / len(accuracies_train)
    # losses_test, accuracies_test = test_epoch(loader_test, state)
    # epoch_loss_test = sum(losses_test) / len(losses_test)
    # epoch_acc_test = sum(accuracies_test) / len(accuracies_test)
    print(f"---\nepoch {i+1} training loss: {epoch_loss_train}; training accuracy: {epoch_acc_train}")
    # print(f"epoch {i+1} test loss: {epoch_loss_test}; test accuracy: {epoch_acc_test}\n---")


