# src/urils/context.py

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
from src.attacks.registry import build_attack


@dataclass
class RunContext:
    # Identity / required core
    cfg: ExperimentConfig
    seed: int
    run_id: str

    # optional training config
    training_cfg: Optional[TrainingConfig] = None

    # runtime core
    device: Optional[torch.device] = None
    model: Optional[nn.Module] = None
    optimizer: Optional[optim.Optimizer] = None
    criterion: Optional[nn.Module] = None
    is_completed: Optional[bool] = None

    # data
    loaders: dict[str, torch.utils.data.DataLoader] = None

    # logging
    logger: Optional[JSONLLogger] = None

    # paths
    run_dir: Optional[Path] = None
    ckpt_dir: Optional[Path] = None
    latest_ckpt: Optional[Path] = None
    final_ckpt: Optional[Path] = None
    best_ckpt: Optional[Path] = None

    # eval-specific extras
    attack_fn: Optional[callable] = None
    attack_params: Optional[dict] = None
    epsilon: Optional[float] = None

    defense_model: Optional[nn.Module] = None

    # state
    epoch: int = 0
    best_acc: float = 0.0


def _base_setup(cfg: ExperimentConfig, seed: int):
    device = get_device()
    set_seed(seed)
    return device


def build_run_id(*, task: str, model: str, dataset: str, seed: int, **kwargs) -> str:
    from src.utils.run_id import make_run_id

    return make_run_id(
        task=task,
        model=model,
        dataset=dataset,
        seed=seed,
        **kwargs,
    )


def build_train_ctx(
    cfg: ExperimentConfig,
    training_cfg: TrainingConfig,
    seed: int,
) -> RunContext:

    device = _base_setup(cfg, seed)

    model = load_or_create_model(cfg.model, device)
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_train_loader(cfg.dataset, training_cfg.batch_size, seed)
    test_loader = get_test_loader(cfg.dataset, training_cfg.batch_size)

    log_path = cfg.paths.logs / "standard.jsonl"
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(str(log_path))

    run_dir = Path("runs") / f"standard_seed{seed}"
    ckpt_dir = cfg.paths.checkpoints / f"standard_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_id = build_run_id(
        task="train",
        model=cfg.model.name,
        dataset=cfg.dataset.name,
        seed=seed,
    )

    return RunContext(
        run_id=run_id,
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


def attack_tag(training_cfg: TrainingConfig) -> str:
    if training_cfg.attack.steps is not None:
        return f"{training_cfg.attack.name}{training_cfg.attack.steps}"
    return training_cfg.attack.name


def build_adv_train_ctx(
    cfg: ExperimentConfig,
    training_cfg: TrainingConfig,
    seed: int,
) -> RunContext:

    device = get_device()
    set_seed(seed)

    model = load_or_create_model(cfg.model, device)
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_train_loader(cfg.dataset, training_cfg.batch_size, seed)
    test_loader = get_test_loader(cfg.dataset, training_cfg.batch_size)

    # ── attack resolution ───────────────────────────────
    # ── attack resolution ───────────────────────────────

    base_attack = training_cfg.attack

    steps = base_attack.steps
    alpha = base_attack.alpha
    epsilon = training_cfg.epsilon

    # resolve alpha
    if training_cfg.attack == "pgd":    
        if alpha == "budget_scaled" or alpha is None:
            if steps is None:
                raise ValueError("Cannot compute alpha without steps")

            alpha = 2.5 * float(epsilon) / float(steps)                 

    attack_cfg = AttackConfig(
        name=base_attack.name,
        epsilon=epsilon,
        steps=steps,
        alpha=alpha,
        restarts=base_attack.restarts,
    )

    attack_fn, attack_params = build_attack(attack_cfg)

    # ── naming / tagging ───────────────────────────────
    tag = attack_tag(training_cfg)

    run_dir = Path("runs") / f"adv_{tag}_seed{seed}"
    ckpt_dir = cfg.paths.checkpoints / f"adv_{tag}_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = cfg.paths.logs / f"adv_{tag}_seed{seed}.jsonl"
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)

    logger = JSONLLogger(str(log_path))


    run_id = build_run_id(
        task="adv_train",
        model=cfg.model.name,
        dataset=cfg.dataset.name,
        attack=tag,
        seed=seed,
    )

    return RunContext(
        run_id=run_id,
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

        # adversarial state
        attack_fn=attack_fn,
        attack_params=attack_params,
        epsilon=epsilon,

        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        latest_ckpt=ckpt_dir / "latest.pth",
        final_ckpt=ckpt_dir / "final.pth",
        best_ckpt=ckpt_dir / "best.pth",
    )


def build_eval_attack_ctx(
    cfg: ExperimentConfig,
    attack_cfg: AttackConfig,
    seed: int,
    epsilon: float,
) -> RunContext:

    device = _base_setup(cfg, seed)

    checkpoint_path = (
        cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"
    )
    model = load_model(str(checkpoint_path), device, cfg.model)

    test_loader = get_test_loader(cfg.dataset, batch_size=64)

    # ---- resolve alpha ----
    steps = attack_cfg.steps
    alpha = attack_cfg.alpha

    if alpha == "budget_scaled":
        alpha = 2.5 * epsilon / steps

    resolved_attack_cfg = AttackConfig(
        name=attack_cfg.name,
        epsilon=epsilon,
        steps=steps,
        alpha=alpha,
        restarts=attack_cfg.restarts,
    )

    attack_fn, attack_params = build_attack(resolved_attack_cfg)

    logger = JSONLLogger(str(cfg.paths.logs / "eval_attack.jsonl"))

    run_id = build_run_id(
        task="eval_attack",
        model=cfg.model.name,
        dataset=cfg.dataset.name,
        attack=attack_cfg.name,
        steps=steps,
        epsilon=epsilon,
        seed=seed,
    )

    return RunContext(
        run_id=run_id,
        cfg=cfg,
        training_cfg=None,
        seed=seed,
        device=device,
        model=model,
        optimizer=None,
        criterion=None,
        loaders={"test": test_loader},
        logger=logger,

        attack_fn=attack_fn,
        attack_params=attack_params,
        epsilon=epsilon,
    )


def build_eval_robustness_ctx(
    cfg: ExperimentConfig,
    training_cfg: TrainingConfig,
    eval_cfg: AttackConfig,
    seed: int,
    epsilon: float,
) -> RunContext:

    device = _base_setup(cfg, seed)

    # ---- models ----
    base_path = cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"
    base_model = load_model(str(base_path), device, cfg.model)

    defense_tag = training_cfg.attack.name + (
        str(training_cfg.attack.steps) if training_cfg.attack.steps else ""
    )

    defense_path = (
        cfg.paths.checkpoints / f"adv_{defense_tag}_seed{seed}" / "final.pth"
    )
    defense_model = load_model(str(defense_path), device, cfg.model)

    # ---- eval attack ----
    steps = eval_cfg.steps
    alpha = eval_cfg.alpha

    if alpha is None and steps is not None:
        alpha = 2.5 * epsilon / steps

    resolved_eval_cfg = AttackConfig(
        name=eval_cfg.name,
        epsilon=epsilon,
        steps=steps,
        alpha=alpha,
        restarts=eval_cfg.restarts,
    )

    eval_attack_fn, eval_attack_params = build_attack(resolved_eval_cfg)

    test_loader = get_test_loader(cfg.dataset, batch_size=64)

    # ---- logging ----
    logger = JSONLLogger(str(cfg.paths.logs / "eval_robustness.jsonl"))

    run_id = build_run_id(
        task="eval_robustness",
        model=cfg.model.name,
        dataset=cfg.dataset.name,
        seed=seed,
        defense=defense_tag,
        eval_attack=eval_cfg.name,
        epsilon=epsilon,
    )

    return RunContext(
        cfg=cfg,
        training_cfg=training_cfg,
        seed=seed,
        device=device,

        model=base_model,
        defense_model=defense_model,

        loaders={"test": test_loader},

        logger=logger,

        attack_fn=eval_attack_fn,
        attack_params=eval_attack_params,

        epsilon=epsilon,
        run_id=run_id,
    )