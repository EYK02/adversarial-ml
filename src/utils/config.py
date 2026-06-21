# src/utils/config.py

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────

@dataclass
class DatasetConfig:
    name:           str
    num_classes:    int
    input_channels: int
    image_size:     int
    data_path:      Path
    mean:           list[float]
    std:            list[float]


@dataclass
class ModelConfig:
    name:           str
    num_classes:    int
    input_channels: int


@dataclass
class AttackConfig:
    name:       str
    steps:      Optional[int]   = None
    alpha:      Optional[float] = None  # None = budget_scaled
    restarts:   int             = 1


@dataclass
class TrainingConfig:
    method:        str
    epochs:        int
    batch_size:    int
    learning_rate: float
    epsilon:       Optional[float] = None   # adversarial only
    attack:        Optional[AttackConfig] = None  # adversarial only


@dataclass
class ExperimentConfig:
    dataset:          DatasetConfig
    model:            ModelConfig
    training:         list[TrainingConfig]
    eval_attacks:     list[AttackConfig]
    seeds:            list[int]
    epsilon_eval:     list[float]
    epsilon_heatmap:  list[float]
    run_name:         str
    experiment_path:  str         

    dry_run:          bool = False
    smoke_test:       bool = False

    # resolved at load time
    paths:            ExperimentPaths = field(default_factory=lambda: None)


@dataclass
class ExperimentPaths:
    run_dir:      Path
    logs:         Path
    checkpoints:  Path
    metrics:      Path
    figures:      Path


# ─────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(path_str: str, root: Path) -> Path:
    """Resolve a path relative to the config root."""
    return (root / path_str).resolve()


def _load_dataset(path_str: str, root: Path) -> DatasetConfig:
    d = _load_yaml(_resolve(path_str, root))
    return DatasetConfig(
        name           = d["name"],
        num_classes    = d["num_classes"],
        input_channels = d["input_channels"],
        image_size     = d["image_size"],
        data_path      = Path(d["data_path"]),
        mean           = d["mean"],
        std            = d["std"],
    )


def _load_model(path_str: str, root: Path) -> ModelConfig:
    d = _load_yaml(_resolve(path_str, root))
    return ModelConfig(
        name           = d["name"],
        num_classes    = d["num_classes"],
        input_channels = d["input_channels"],
    )


def _load_attack(path_str: str, root: Path) -> AttackConfig:
    d = _load_yaml(_resolve(path_str, root))
    return AttackConfig(
        name        = d["name"],
        steps       = d.get("steps"),
        alpha       = None if d.get("alpha") == "budget_scaled" else d.get("alpha"),
        restarts    = d.get("restarts", 1)
    )


def _load_training(path_str: str, root: Path) -> TrainingConfig:
    d = _load_yaml(_resolve(path_str, root))
    attack = None
    if "attack" in d:
        attack = _load_attack(d["attack"], root)
    return TrainingConfig(
        method        = d["method"],
        epochs        = d["epochs"],
        batch_size    = d["batch_size"],
        learning_rate = d["learning_rate"],
        epsilon       = d.get("epsilon"),
        attack        = attack,
    )


def _make_paths(run_dir: Path, cfg_dataset: str, cfg_model: str) -> ExperimentPaths:
    shared_checkpoints = Path("checkpoints") / cfg_dataset / cfg_model

    return ExperimentPaths(
        run_dir     = run_dir,
        logs        = run_dir / "logs",
        checkpoints = shared_checkpoints,
        metrics     = run_dir / "metrics",
        figures     = run_dir / "figures",
    )


def _resolve_run_name(exp_path: Path, dry_run: bool, smoke_test: bool) -> str:
    from datetime import datetime
    stem = exp_path.stem
    tag  = "_smoke" if smoke_test else "_dry" if dry_run else ""
    return f"{datetime.now().strftime('%Y-%m-%d')}_{stem}{tag}"


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────

def load_experiment(
        path: str | Path, 
        dry_run: bool = False,
        smoke_test: bool = False
    ) -> ExperimentConfig:
    
    path = Path(path).resolve()
    root = path.parent.parent.parent  # configs/ root — one level above experiments/
    raw  = _load_yaml(path)

    # apply override runs
    if smoke_test and "smoke_test" in raw:
        for key, val in raw["smoke_test"].items():
            raw[key] = val
    elif dry_run and "dry_run" in raw:
        overrides = raw["dry_run"]
        for key, val in overrides.items():
            raw[key] = val

    training = [_load_training(t, root) for t in raw["training"]]
    eval_attacks = [_load_attack(a, root) for a in raw["eval_attacks"]]

    dataset         = _load_dataset(raw["dataset"], root)
    model           = _load_model(raw["model"], root)

    run_name = _resolve_run_name(path, dry_run, smoke_test)
    run_dir  = Path("runs") / run_name
    paths    = _make_paths(run_dir, dataset.name, model.name)

    cfg = ExperimentConfig(
        dataset         = dataset,
        model           = model,
        training        = training,
        eval_attacks    = eval_attacks,
        seeds           = raw["seeds"],
        epsilon_eval    = raw["epsilon_eval"],
        epsilon_heatmap = raw["epsilon_heatmap"],
        run_name        = run_name,
        experiment_path = str(path),
        dry_run         = dry_run,
        smoke_test      = smoke_test,
        paths           = paths,
    )

    _validate(cfg)
    return cfg


def _validate(cfg: ExperimentConfig):
    for t in cfg.training:
        if t.method == "adversarial":
            assert t.epsilon is not None, \
                f"Training config '{t.method}' missing epsilon"
            assert t.attack is not None, \
                f"Adversarial training config missing attack"
    assert len(cfg.seeds) > 0, "No seeds specified"
    assert len(cfg.epsilon_eval) > 0, "No eval epsilons specified"