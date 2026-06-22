# src/runner/utils

from pathlib import Path

import torch

from src.utils.config import AttackConfig, ExperimentConfig, TrainingConfig
from src.utils.logger import JSONLLogger
from src.utils.seed import set_seed, get_device


def _setup(cfg: ExperimentConfig, seed: int) -> torch.device:
    set_seed(seed)
    return get_device()


def _make_logger(path: Path) -> JSONLLogger:
    path.parent.mkdir(parents=True, exist_ok=True)
    return JSONLLogger(str(path))


def _make_ckpt_dir(base: Path, tag: str) -> Path:
    d = base / tags
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
    from src.runner.run_id import make_run_id
    return make_run_id(task=task, model=model, dataset=dataset, seed=seed, **kwargs)


def attack_tag(training_cfg: TrainingConfig) -> str:
    if training_cfg.attack.steps is not None:
        return f"{training_cfg.attack.name}{training_cfg.attack.steps}"
    return training_cfg.attack.name
