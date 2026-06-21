# src/utils/seed.py

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")