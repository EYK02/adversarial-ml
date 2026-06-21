from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.seed import set_seed, get_device
from src.utils.logger import JSONLLogger
from src.models.factory import load_or_create_model, load_model
from src.datasets.mnist import get_train_loader, get_test_loader
from src.utils.config import ExperimentConfig, TrainingConfig, AttackConfig

@dataclass
class RunContext:
    # config
    cfg: ExperimentConfig
    training_cfg: Optional[TrainingConfig]
    seed: int

    # runtime core
    device: torch.device
    model: nn.Module
    optimizer: Optional[optim.Optimizer]
    criterion: Optional[nn.Module]

    # data
    loaders: dict[str, torch.utils.data.DataLoader]

    # logging
    logger: JSONLLogger

    # paths
    run_dir: Optional[Path] = None
    ckpt_dir: Optional[Path] = None
    latest_ckpt: Optional[Path] = None
    final_ckpt: Optional[Path] = None
    best_ckpt: Optional[Path] = None

    # extra (evaluation-only)
    defense_model: Optional[nn.Module] = None

    # state
    epoch: int = 0
    best_acc: float = 0.0

def _base_setup(cfg: ExperimentConfig, seed: int):
    device = get_device()
    set_seed(seed)
    return device

def build_train_ctx(
    cfg: ExperimentConfig,
    training_cfg: TrainingConfig,
    seed: int,
    logger: JSONLLogger,
) -> RunContext:

    device = _base_setup(cfg, seed)

    model = load_or_create_model(cfg.model, device)
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_train_loader(cfg.dataset, training_cfg.batch_size, seed)
    test_loader = get_test_loader(cfg.dataset, training_cfg.batch_size)

    run_dir = Path("runs") / f"standard_seed{seed}"
    ckpt_dir = cfg.paths.checkpoints / f"standard_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        cfg=cfg,
        training_cfg=training_cfg,
        seed=seed,
        device=device,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        loaders={
            "train": train_loader,
            "test": test_loader,
        },
        logger=logger,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        latest_ckpt=ckpt_dir / "latest.pth",
        final_ckpt=ckpt_dir / "final.pth",
        best_ckpt=ckpt_dir / "best.pth",
    )

def build_eval_attack_ctx(
    cfg: ExperimentConfig,
    seed: int,
    logger: JSONLLogger,
) -> RunContext:

    device = _base_setup(cfg, seed)

    checkpoint_path = cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"
    model = load_model(str(checkpoint_path), device, cfg.model)

    test_loader = get_test_loader(cfg.dataset, batch_size=64)

    return RunContext(
        cfg=cfg,
        training_cfg=None,
        seed=seed,
        device=device,
        model=model,
        optimizer=None,
        criterion=None,
        loaders={"test": test_loader},
        logger=logger,
    )

def build_eval_robustness_ctx(
    cfg: ExperimentConfig,
    training_cfg: TrainingConfig,
    seed: int,
    logger: JSONLLogger,
) -> RunContext:

    device = _base_setup(cfg, seed)

    # baseline model
    base_path = cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"
    base_model = load_model(str(base_path), device, cfg.model)

    # defense model
    defense_tag = training_cfg.attack.name + (
        str(training_cfg.attack.steps) if training_cfg.attack.steps else ""
    )

    defense_path = cfg.paths.checkpoints / f"adv_{defense_tag}_seed{seed}" / "final.pth"
    defense_model = load_model(str(defense_path), device, cfg.model)

    test_loader = get_test_loader(cfg.dataset, batch_size=64)

    return RunContext(
        cfg=cfg,
        training_cfg=training_cfg,
        seed=seed,
        device=device,
        model=base_model,
        defense_model=defense_model,
        optimizer=None,
        criterion=None,
        loaders={"test": test_loader},
        logger=logger,
    )