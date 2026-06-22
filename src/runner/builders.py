# src/runner/builders.py

from pathlib import Path
import sys
from typing import List, Tuple

import torch

from src.attacks.registry import build_attack
from src.datasets.mnist import get_test_loader, get_train_loader
from src.models.factory import load_model, load_or_create_model
from src.runner.context import RunContext
from src.runner.executor import Experiment
from src.runner.utils import _ckpt_paths, _make_ckpt_dir, _make_logger, _resolve_alpha, _setup, attack_tag, build_run_id
from src.utils.config import AttackConfig, ExperimentConfig, TrainingConfig



def build_experiments(
    cfg: ExperimentConfig,
    dry_run: bool = False,
    smoke_test: bool = False,
    run_name: str | None = None,
) -> List[Tuple[str, List[Experiment]]]:
    """
    Pure function:
    defines experiment graph (stages → experiments).
    """

    mode_flags = []

    if dry_run:
        mode_flags += ["--dry-run"]
    if smoke_test:
        mode_flags += ["--smoke-test"]
    if run_name:
        mode_flags += ["--run-name", run_name]

    py = sys.executable
    exp_path = cfg.experiment_path

    stages: list[tuple[str, list[Experiment]]] = []

    # ─────────────────────────────────────────────
    # Stage 1: Standard training
    # ─────────────────────────────────────────────
    stage1 = [
        Experiment(
            name=f"standard train seed={seed}",
            command=[
                py, "-m", "src.training.standard",
                "--experiment", exp_path,
                "--seed", str(seed),
                *mode_flags,
            ],
        )
        for seed in cfg.seeds
    ]
    stages.append(("STAGE 1 — Standard training", stage1))

    # ─────────────────────────────────────────────
    # Stage 2: Attack evaluation
    # ─────────────────────────────────────────────
    stage2 = [
        Experiment(
            name=f"attack eval {a.name}{a.steps or ''} seed={seed}",
            command=[
                py, "-m", "src.evaluation.eval_attack",
                "--experiment", exp_path,
                "--attack", f"{a.name}{a.steps or ''}",
                "--seed", str(seed),
                *mode_flags,
            ],
        )
        for seed in cfg.seeds
        for a in cfg.eval_attacks
    ]
    stages.append(("STAGE 2 — Attack evaluation", stage2))

    # ─────────────────────────────────────────────
    # Stage 3: Adversarial training
    # ─────────────────────────────────────────────
    stage3 = [
        Experiment(
            name=f"adv train {attack_tag(t)} seed={seed}",
            command=[
                py, "-m", "src.training.adversarial",
                "--experiment", exp_path,
                "--training-config", attack_tag(t),
                "--seed", str(seed),
                *mode_flags,
            ],
        )
        for t in cfg.training
        if t.method == "adversarial"
        for seed in cfg.seeds
    ]
    stages.append(("STAGE 3 — Adversarial training", stage3))

    # ─────────────────────────────────────────────
    # Stage 4: Robustness evaluation
    # ─────────────────────────────────────────────
    stage4 = [
        Experiment(
            name=f"robustness eval def={attack_tag(t)} eval={a.name}{a.steps or ''} seed={seed}",
            command=[
                py, "-m", "src.evaluation.eval_robustness",
                "--experiment", exp_path,
                "--training-config", attack_tag(t),
                "--eval-attack", f"{a.name}{a.steps or ''}",
                "--seed", str(seed),
                *mode_flags,
            ],
        )
        for t in cfg.training
        if t.method == "adversarial"
        for a in cfg.eval_attacks
        for seed in cfg.seeds
    ]
    stages.append(("STAGE 4 — Robustness evaluation", stage4))

    # ─────────────────────────────────────────────
    # Stage 5: Analysis
    # ─────────────────────────────────────────────
    stage5 = [
        Experiment(
            name="analysis report",
            command=[
                py, "-m", "src.analysis.report",
                "--experiment", exp_path,
                *mode_flags,
            ],
        )
    ]
    stages.append(("STAGE 5 — Analysis report", stage5))

    return stages

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
        optimizer=torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate),
        criterion=torch.nn.CrossEntropyLoss(),
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
        optimizer=torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate),
        criterion=torch.nn.CrossEntropyLoss(),
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