# src/runner/context.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src.utils.config import ExperimentConfig, TrainingConfig
from src.utils.logger import JSONLLogger


@dataclass
class RunContext:
    # Identity
    cfg:          ExperimentConfig
    seed:         int
    run_id:       str

    # optional training config
    training_cfg: Optional[TrainingConfig] = None

    # runtime core
    device:       Optional[torch.device]       = None
    model:        Optional[torch.nn.Module]          = None
    optimizer:    Optional[torch.optim.Optimizer]    = None
    criterion:    Optional[torch.nn.Module]          = None
    is_completed: Optional[bool]               = None

    # data
    loaders: dict[str, torch.utils.data.DataLoader] = None

    # logging
    logger: Optional[JSONLLogger] = None

    # paths
    run_dir:     Optional[Path] = None
    ckpt_dir:    Optional[Path] = None
    latest_ckpt: Optional[Path] = None
    final_ckpt:  Optional[Path] = None
    best_ckpt:   Optional[Path] = None

    # eval-specific
    attack_fn:     Optional[callable] = None
    attack_params: Optional[dict]     = None
    epsilon:       Optional[float]    = None
    defense_model: Optional[torch.nn.Module] = None

    # state
    epoch:    int   = 0
    best_acc: float = 0.0
