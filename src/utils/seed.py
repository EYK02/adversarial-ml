# src/utils/reproducibility.py

import torch
import random
import numpy as np
import os

def set_seed(seed):
    torch.set_float32_matmul_precision('high')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # required for deterministic CUDA math
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")