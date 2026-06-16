# utils/data.py

from random import random
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_mnist_train_loader(batch_size=64, seed=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=False,
        transform=transform
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=0
    )

def get_mnist_test_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )