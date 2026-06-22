# src/runner/context.py

"""
RunContext definition.

RunContext is the central runtime state container for a single experiment.
It aggregates configuration, model state, data loaders, logging, and
optional attack/robustness components.

It is passed through training, evaluation, and analysis pipelines.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from src.utils.config import ExperimentConfig, TrainingConfig
from src.utils.logger import JSONLLogger


@dataclass
class RunContext:
    """
    Container for all runtime state of an experiment.

    Attributes
    ----------
    cfg:
        Global experiment configuration.

    training_cfg:
        Optional training configuration.

    seed:
        Random seed used for reproducibility.

    run_id:
        Unique identifier for this run.

    device:
        Computation device (CPU/GPU).

    model:
        Primary model under evaluation or training.

    optimizer:
        Optimizer used during training (if applicable).

    criterion:
        Loss function.

    loaders:
        Dictionary of DataLoader objects (train/test/etc).

    logger:
        JSONL logger for metrics and events.

    run_dir:
        Root directory for this run’s outputs.

    ckpt_dir:
        Checkpoint directory.

    attack_fn:
        Optional adversarial attack function.

    attack_params:
        Resolved attack parameters.

    epsilon:
        Perturbation budget (if applicable).

    defense_model:
        Optional secondary model (e.g., adversarially trained model).'

    epoch:
        Current epoch in training.

    best_cc:
        Best performing version in terms of accuracy.

    Notes
    -----
    RunContext is mutable during execution and is passed between pipeline stages.
    """
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
