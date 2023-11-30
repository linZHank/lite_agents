import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets


# Download training data from open datasets.
data_dir = sys.path[0] + '/data'
print(data_dir)
training_data = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
)
