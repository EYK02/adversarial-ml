# src/utils/context.py

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.mnist import get_train_loader, get_test_loader
from src.models.factory import load_or_create_model
from src.utils.seed import get_device
from src.utils.logger import JSONLLogger
from src.utils.config import ExperimentConfig, TrainingConfig

@dataclass
class RunContext:
    # config
    cfg: ExperimentConfig
    training_cfg: TrainingConfig | None
    seed: int

    # runtime core
    device: torch.device
    model: nn.Module
    optimizer: optim.Optimizer | None
    criterion: nn.Module

    # data loaders
    loaders: dict[str, torch.utils.data.DataLoader]

    # attacks
    attack_fn: callable | None = None
    attack_params: dict | None = None
    epsilon: float | None = None

    # logging
    logger: JSONLLogger | None = None

    # paths
    run_dir: Path | None = None
    ckpt_dir: Path | None = None
    latest_ckpt: Path | None = None
    final_ckpt: Path | None = None
    best_ckpt: Path | None = None

    # state
    epoch: int = 0
    best_acc: float = 0.0

def build_ctx(cfg, training_cfg, seed: int) -> RunContext:
    device = get_device()

    # ── paths ─────────────────────────────────────────────
    run_ckpt_dir = cfg.paths.checkpoints / f"{training_cfg.method}_seed{seed}"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_dir = cfg.paths.run_dir if hasattr(cfg.paths, "run_dir") else None

    latest_ckpt = run_ckpt_dir / "latest.pth"
    final_ckpt  = run_ckpt_dir / "final.pth"
    best_ckpt   = run_ckpt_dir / "best.pth"

    # ── early skip (completed runs) ───────────────────────
    if final_ckpt.exists():
        print(f"[SKIP] {training_cfg.method} seed={seed} already completed.")
        return None

    # ── data ──────────────────────────────────────────────
    train_loader = get_train_loader(cfg.dataset, training_cfg.batch_size, seed)
    test_loader  = get_test_loader(cfg.dataset, training_cfg.batch_size)
    loaders={
        "train": train_loader,
        "test": test_loader,
    }


    # ── model / optimizer ────────────────────────────────
    model = load_or_create_model(cfg.model, device)
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)

    # ── loss ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── logger ────────────────────────────────────────────
    log_path = cfg.paths.logs / f"{training_cfg.method}.jsonl"
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(str(log_path))

    return RunContext(
        cfg=cfg,
        training_cfg=training_cfg,
        seed=seed,

        device=device,
        model=model,
        optimizer=optimizer,
        criterion=criterion,

        loaders=loaders,

        logger=logger,

        run_dir=run_dir,
        ckpt_dir=run_ckpt_dir,
        latest_ckpt=latest_ckpt,
        final_ckpt=final_ckpt,
        best_ckpt=best_ckpt,

        epoch=0,
        best_acc=0.0,
    )