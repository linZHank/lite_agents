import pytest

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import jax.numpy as jnp
from flax import nnx  # The Flax NNX API.
from functools import partial

import matplotlib.pyplot as plt

# Hyper-parameters
BATCH_SIZE = 64
NUM_EPOCHS = 5
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


class CNN(nnx.Module):
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
model = CNN(rngs=nnx.Rngs(0))
# Visualize it.
nnx.display(model)
