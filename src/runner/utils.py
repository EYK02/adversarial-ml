# src/runner/utils

from pathlib import Path

import torch

from src.utils.config import AttackConfig, ExperimentConfig, TrainingConfig
from src.utils.logger import JSONLLogger
from src.utils.seed import set_seed, get_device


def setup(cfg: ExperimentConfig, seed: int) -> torch.device:
    set_seed(seed)
    return get_device()


def make_logger(path: Path) -> JSONLLogger:
    path.parent.mkdir(parents=True, exist_ok=True)
    return JSONLLogger(str(path))


def make_ckpt_dir(base: Path, tag: str) -> Path:
    d = base / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def ckpt_paths(ckpt_dir: Path) -> tuple[Path, Path, Path]:
    """Returns (latest, final, best)."""
    return ckpt_dir / "latest.pth", ckpt_dir / "final.pth", ckpt_dir / "best.pth"


def build_run_id(*, task: str, model: str, dataset: str, seed: int, **kwargs) -> str:
    from src.runner.run_id import make_run_id
    return make_run_id(task=task, model=model, dataset=dataset, seed=seed, **kwargs)