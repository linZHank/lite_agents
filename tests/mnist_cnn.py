import pytest
from functools import partial

from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from jax import random
import jax.numpy as jnp
from flax import nnx  # The Flax NNX API.
import optax

import matplotlib.pyplot as plt

# Hyper-parameters
BATCH_SIZE = 64
MAX_EPOCHS = 2
LEARNING_RATE = 3e-4
MOMENTUM = 0.9
SEED = 19

# Create transforms for data
torch_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.1307,), (0.3081,)),
    ]
)

# Download Datsets
train_dataset = datasets.MNIST(
    # sys.path[0] + "/data",
    Path(__file__).parent.joinpath("data"),
    train=True,
    download=True,
    transform=torch_transform,
)
test_dataset = datasets.MNIST(
    # sys.path[0] + "/data",
    Path(__file__).parent.joinpath("data"),
    train=False,
    download=True,
    transform=torch_transform,
)

# Create dataloaders
loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Check data
sample_x, sample_y = next(iter(loader_train))
jsx = jnp.array(sample_x.numpy())
jsy = jnp.array(sample_y.numpy())
print(jsx.shape)  # 64x1x28x28
print(jsy.shape)  # 64
print(jsy)

# Create a CNN


class VanillaConvNet(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Instantiate the model.
model = VanillaConvNet(rngs=nnx.Rngs(0))
# Visualize it.
nnx.display(model)
# Sanity test
dummy_x = random.uniform(key=random.key(0), shape=(3, 28, 28, 1))
# print(dummy_x.shape)
dummy_preds = model(dummy_x)
print(dummy_preds)

# Create optimizer

optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, MOMENTUM))
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
)

nnx.display(optimizer)


# Define training steps
def loss_fn(
    model: VanillaConvNet, features_array: np.ndarray, labels_array: np.ndarray
):
    logits = model(features_array)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels_array
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: VanillaConvNet,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch_features: np.ndarray,
    batch_labels: np.ndarray,
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch_features, batch_labels)
    metrics.update(loss=loss, logits=logits, labels=batch_labels)  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(
    model: VanillaConvNet,
    metrics: nnx.MultiMetric,
    batch_features: np.ndarray,
    batch_labels: np.ndarray,
):
    loss, logits = loss_fn(model, batch_features, batch_labels)
    metrics.update(loss=loss, logits=logits, labels=batch_labels)  # In-place updates.


# Train model
metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

for ep in range(MAX_EPOCHS):
    # train 1 epoch
    for i, train_batch in enumerate(loader_train):
        features_train = torch.permute(train_batch[0], (0, 2, 3, 1)).numpy()
        labels_train = train_batch[1].numpy()
        train_step(
            model,
            optimizer,
            metrics,
            features_train,
            labels_train,
        )
        print(
            f"batch: {i + 1} loss: {metrics.compute()['loss']}, accuracy: {metrics.compute()['accuracy']}"
        )
    # Log training metrics
    for metric, value in metrics.compute().items():  # Compute the metrics.
        metrics_history[f"train_{metric}"].append(value)  # Record the metrics.
    metrics.reset()  # Reset the metrics for the test set.
    # Evaluate 1 epoch
    for i, eval_batch in enumerate(loader_test):
        features_eval = torch.permute(eval_batch[0], (0, 2, 3, 1)).numpy()
        labels_eval = eval_batch[1].numpy()
        eval_step(model, metrics, features_eval, labels_eval)
    # Log the test metrics.
    for metric, value in metrics.compute().items():
        metrics_history[f"test_{metric}"].append(value)
    metrics.reset()  # Reset the metrics for the next training epoch.

    print(f"Epoch {ep} metrics: {metrics_history}")


# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title("Loss")
ax2.set_title("Accuracy")
for dataset in ("train", "test"):
    ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
    ax2.plot(metrics_history[f"{dataset}_accuracy"], label=f"{dataset}_accuracy")
ax1.legend()
ax2.legend()
plt.show()
