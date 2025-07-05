import pytest

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import jax.numpy as jnp

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


