# src/datasets/mnist.py

import numpy as np
from random import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.config import DatasetConfig


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _get_transform(cfg: DatasetConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])


def get_mnist_train_loader(cfg: DatasetConfig, batch_size: int, seed: int) -> DataLoader:
    transform = _get_transform(cfg)

    dataset = datasets.MNIST(
        root=str(cfg.data_path),
        train=True,
        download=True,
        transform=transform,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=_seed_worker,
        num_workers=4,
        pin_memory=True
    )


def get_mnist_test_loader(cfg: DatasetConfig, batch_size: int) -> DataLoader:
    transform = _get_transform(cfg)

    dataset = datasets.MNIST(
        root=str(cfg.data_path),
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )