# src/datasets/factory.py

from torch.utils.data import DataLoader

from src.utils.config import DatasetConfig
from src.datasets.mnist import get_mnist_test_loader, get_mnist_train_loader


def get_train_loader(
    cfg: DatasetConfig,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """
    Build a training dataloader for the configured dataset.
    """

    if cfg.name.lower() == "mnist":
        return get_mnist_train_loader(cfg, batch_size, seed)
    
    # elif cfg.name.lower() == "cifar10":
    #     return get_mnist_train_loader(cfg, batch_size, seed)
    
    raise ValueError(f"Unsupported dataset: {cfg.name}")


def get_test_loader(
    cfg: DatasetConfig,
    batch_size: int,
) -> DataLoader:
    """
    Build a test dataloader for the configured dataset.
    """

    if cfg.name.lower() == "mnist":
        return get_mnist_test_loader(cfg, batch_size)

    # elif cfg.name.lower() == "cifar10":
    #     return get_mnist_test_loader(cfg, batch_size)

    raise ValueError(f"Unsupported dataset: {cfg.name}")