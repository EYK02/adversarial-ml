# src/utils/context.py

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
    # Identity
    cfg:          ExperimentConfig
    seed:         int
    run_id:       str

    # optional training config
    training_cfg: Optional[TrainingConfig] = None

    # runtime core
    device:       Optional[torch.device]       = None
    model:        Optional[nn.Module]          = None
    optimizer:    Optional[optim.Optimizer]    = None
    criterion:    Optional[nn.Module]          = None
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
    defense_model: Optional[nn.Module] = None

    # state
    epoch:    int   = 0
    best_acc: float = 0.0


# ── internal helpers ──────────────────────────────────────────────────────────

def _setup(cfg: ExperimentConfig, seed: int) -> torch.device:
    set_seed(seed)
    return get_device()


def _make_logger(path: Path) -> JSONLLogger:
    path.parent.mkdir(parents=True, exist_ok=True)
    return JSONLLogger(str(path))


def _make_ckpt_dir(base: Path, tag: str) -> Path:
    d = base / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ckpt_paths(ckpt_dir: Path) -> tuple[Path, Path, Path]:
    """Returns (latest, final, best)."""
    return ckpt_dir / "latest.pth", ckpt_dir / "final.pth", ckpt_dir / "best.pth"


def _resolve_alpha(attack_cfg: AttackConfig, epsilon: float) -> float:
    alpha = attack_cfg.alpha
    if attack_cfg.name == "pgd" and (alpha == "budget_scaled" or alpha is None):
        if attack_cfg.steps is None:
            raise ValueError("Cannot compute alpha without steps")
        alpha = 2.5 * float(epsilon) / float(attack_cfg.steps)
    return alpha


def build_run_id(*, task: str, model: str, dataset: str, seed: int, **kwargs) -> str:
    from src.utils.run_id import make_run_id
    return make_run_id(task=task, model=model, dataset=dataset, seed=seed, **kwargs)


def attack_tag(training_cfg: TrainingConfig) -> str:
    if training_cfg.attack.steps is not None:
        return f"{training_cfg.attack.name}{training_cfg.attack.steps}"
    return training_cfg.attack.name


# ── public builders ───────────────────────────────────────────────────────────
def build_train_ctx(
    cfg:          ExperimentConfig,
    training_cfg: TrainingConfig,
    seed:         int,
) -> RunContext:

    device   = _setup(cfg, seed)
    model    = load_or_create_model(cfg.model, device)
    ckpt_dir = _make_ckpt_dir(cfg.paths.checkpoints, f"standard_seed{seed}")
    latest, final, best = _ckpt_paths(ckpt_dir)

    return RunContext(
        run_id=build_run_id(
            task="train", model=cfg.model.name, dataset=cfg.dataset.name, seed=seed,
        ),
        cfg=cfg,
        training_cfg=training_cfg,
        seed=seed,
        device=device,
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=training_cfg.learning_rate),
        criterion=nn.CrossEntropyLoss(),
        loaders={
            "train": get_train_loader(cfg.dataset, training_cfg.batch_size, seed),
            "test":  get_test_loader(cfg.dataset, training_cfg.batch_size),
        },
        logger=_make_logger(cfg.paths.logs / "standard.jsonl"),
        run_dir=Path("runs") / f"standard_seed{seed}",
        ckpt_dir=ckpt_dir,
        latest_ckpt=latest,
        final_ckpt=final,
        best_ckpt=best,
    )


def build_adv_train_ctx(
    cfg:          ExperimentConfig,
    training_cfg: TrainingConfig,
    seed:         int,
) -> RunContext:

    device  = _setup(cfg, seed)
    model   = load_or_create_model(cfg.model, device)
    tag     = attack_tag(training_cfg)
    epsilon = training_cfg.epsilon

    alpha = _resolve_alpha(training_cfg.attack, epsilon)  # ← fixes the string-compare bug

    attack_cfg = AttackConfig(
        name=training_cfg.attack.name,
        epsilon=epsilon,
        steps=training_cfg.attack.steps,
        alpha=alpha,
        restarts=training_cfg.attack.restarts,
    )
    attack_fn, attack_params = build_attack(attack_cfg)

    ckpt_dir = _make_ckpt_dir(cfg.paths.checkpoints, f"adv_{tag}_seed{seed}")
    latest, final, best = _ckpt_paths(ckpt_dir)

    return RunContext(
        run_id=build_run_id(
            task="adv_train", model=cfg.model.name, dataset=cfg.dataset.name,
            attack=tag, seed=seed,
        ),
        cfg=cfg,
        training_cfg=training_cfg,
        seed=seed,
        device=device,
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=training_cfg.learning_rate),
        criterion=nn.CrossEntropyLoss(),
        loaders={
            "train": get_train_loader(cfg.dataset, training_cfg.batch_size, seed),
            "test":  get_test_loader(cfg.dataset, training_cfg.batch_size),
        },
        logger=_make_logger(cfg.paths.logs / f"adv_{tag}_seed{seed}.jsonl"),
        run_dir=Path("runs") / f"adv_{tag}_seed{seed}",
        ckpt_dir=ckpt_dir,
        latest_ckpt=latest,
        final_ckpt=final,
        best_ckpt=best,
        attack_fn=attack_fn,
        attack_params=attack_params,
        epsilon=epsilon,
    )


def build_eval_attack_ctx(
    cfg:        ExperimentConfig,
    attack_cfg: AttackConfig,
    seed:       int,
    epsilon:    float,
) -> RunContext:

    device = _setup(cfg, seed)
    model  = load_model(
        str(cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"),
        device, cfg.model,
    )

    alpha = _resolve_alpha(attack_cfg, epsilon)

    resolved = AttackConfig(
        name=attack_cfg.name,
        epsilon=epsilon,
        steps=attack_cfg.steps,
        alpha=alpha,
        restarts=attack_cfg.restarts,
    )
    attack_fn, attack_params = build_attack(resolved)

    return RunContext(
        run_id=build_run_id(
            task="eval_attack", model=cfg.model.name, dataset=cfg.dataset.name,
            attack=attack_cfg.name, steps=attack_cfg.steps, epsilon=epsilon, seed=seed,
        ),
        cfg=cfg,
        seed=seed,
        device=device,
        model=model,
        loaders={"test": get_test_loader(cfg.dataset, batch_size=64)},
        logger=_make_logger(cfg.paths.logs / "eval_attack.jsonl"),
        attack_fn=attack_fn,
        attack_params=attack_params,
        epsilon=epsilon,
    )


def build_eval_robustness_ctx(
    cfg:          ExperimentConfig,
    training_cfg: TrainingConfig,
    eval_cfg:     AttackConfig,
    seed:         int,
    epsilon:      float,
) -> RunContext:

    device       = _setup(cfg, seed)
    defense_tag  = attack_tag(training_cfg)

    base_model    = load_model(
        str(cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"),
        device, cfg.model,
    )
    defense_model = load_model(
        str(cfg.paths.checkpoints / f"adv_{defense_tag}_seed{seed}" / "final.pth"),
        device, cfg.model,
    )

    alpha = _resolve_alpha(eval_cfg, epsilon)

    resolved = AttackConfig(
        name=eval_cfg.name,
        epsilon=epsilon,
        steps=eval_cfg.steps,
        alpha=alpha,
        restarts=eval_cfg.restarts,
    )
    eval_attack_fn, eval_attack_params = build_attack(resolved)

    return RunContext(
        run_id=build_run_id(
            task="eval_robustness", model=cfg.model.name, dataset=cfg.dataset.name,
            seed=seed, defense=defense_tag, eval_attack=eval_cfg.name, epsilon=epsilon,
        ),
        cfg=cfg,
        training_cfg=training_cfg,
        seed=seed,
        device=device,
        model=base_model,
        defense_model=defense_model,
        loaders={"test": get_test_loader(cfg.dataset, batch_size=64)},
        logger=_make_logger(cfg.paths.logs / "eval_robustness.jsonl"),
        attack_fn=eval_attack_fn,
        attack_params=eval_attack_params,
        epsilon=epsilon,
    )