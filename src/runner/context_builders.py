# src/runner/context_builders.py

"""
RunContext builders.

This module contains factory functions that construct fully initialized
RunContext objects for different experiment modes:

- standard training
- adversarial training
- attack evaluation
- robustness evaluation

Builders are responsible for wiring together:
models, datasets, optimizers, attacks, logging, and filesystem paths.
"""

from pathlib import Path

import torch

from src.attacks.registry import build_attack, attack_tag
from src.datasets.factory import get_test_loader, get_train_loader
from src.models.factory import load_model, load_or_create_model
from src.runner.context import RunContext
from src.runner.utils import ckpt_paths, make_ckpt_dir, make_logger, setup, build_run_id
from src.utils.config import AttackConfig, ExperimentConfig, TrainingConfig


def with_epsilon(
    attack_cfg: AttackConfig,
    epsilon: float,
) -> AttackConfig:
    return AttackConfig(
        method=attack_cfg.method,
        epsilon=epsilon,
        steps=attack_cfg.steps,
        alpha=attack_cfg.alpha,
        restarts=attack_cfg.restarts,
    )


def build_train_ctx(
    cfg:          ExperimentConfig,
    training_cfg: TrainingConfig,
    seed:         int,
) -> RunContext:
    """
    Construct RunContext for standard (non-adversarial) training.

    Returns a fully initialized context containing:
    - model
    - optimizer
    - loss function
    - training and test loaders
    - checkpoint paths
    - logger

    No attack components are included.
    """
    device   = setup(cfg, seed)
    model    = load_or_create_model(cfg.model, device)
    ckpt_dir = make_ckpt_dir(cfg.paths.checkpoints, f"standard_seed{seed}")
    latest, final, best = ckpt_paths(ckpt_dir)

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
        logger=make_logger(cfg.paths.logs / "standard.jsonl"),
        run_dir=Path("runs") / f"standard_seed{seed}",
        ckpt_dir=ckpt_dir,
        latest_ckpt=latest,
        final_ckpt=final,
        best_ckpt=best,
    )


def build_adv_train_ctx(
    cfg: ExperimentConfig,
    training_cfg: TrainingConfig,
    seed: int,
) -> RunContext:
    """
    Construct RunContext for adversarial training.

    Includes:
    - standard training components (model, optimizer, loaders)
    - attack function for on-the-fly adversarial example generation
    - attack parameters (epsilon, alpha, steps, etc.)

    Supports automatic resume from latest checkpoint.
    """

    device = setup(cfg, seed)

    model = load_or_create_model(cfg.model, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.learning_rate,
    )

    tag = attack_tag(training_cfg.attack)
    epsilon = training_cfg.epsilon

    attack_fn, attack_params = build_attack(
        with_epsilon(training_cfg.attack, epsilon)
    )

    ckpt_dir = make_ckpt_dir(
        cfg.paths.checkpoints,
        f"adv_{tag}_seed{seed}",
    )

    latest, final, best = ckpt_paths(ckpt_dir)

    # --------------------------------------------------
    # Resume support
    # --------------------------------------------------

    start_epoch = 0
    best_acc = 0.0

    if latest.exists() and not final.exists():

        checkpoint = torch.load(
            latest,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_test_acc"]

        print(
            f"[RESUME] adv_{tag}_seed{seed} "
            f"from epoch {start_epoch}"
        )

    return RunContext(
        run_id=build_run_id(
            task="adv_train",
            model=cfg.model.name,
            dataset=cfg.dataset.name,
            attack=tag,
            seed=seed,
        ),
        cfg=cfg,
        training_cfg=training_cfg,
        seed=seed,
        device=device,
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        loaders={
            "train": get_train_loader(
                cfg.dataset,
                training_cfg.batch_size,
                seed,
            ),
            "test": get_test_loader(
                cfg.dataset,
                training_cfg.batch_size,
            ),
        },
        logger=make_logger(
            cfg.paths.logs / f"adv_{tag}_seed{seed}.jsonl"
        ),
        run_dir=Path("runs") / f"adv_{tag}_seed{seed}",
        ckpt_dir=ckpt_dir,
        latest_ckpt=latest,
        final_ckpt=final,
        best_ckpt=best,
        attack_fn=attack_fn,
        attack_params=attack_params,
        epsilon=epsilon,
        epoch=start_epoch,
        best_acc=best_acc,
    )


def build_eval_attack_ctx(
    cfg:        ExperimentConfig,
    attack_cfg: AttackConfig,
    seed:       int,
    epsilon:    float,
) -> RunContext:
    """
    Construct RunContext for evaluating model robustness under attack.

    Loads a pretrained model and attaches an attack function
    for inference-time adversarial evaluation.

    No training components are initialized.
    """
    device = setup(cfg, seed)
    model  = load_model(
        str(cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"),
        device, cfg.model,
    )

    attack_fn, attack_params = build_attack(
        with_epsilon(attack_cfg, epsilon)
    )

    return RunContext(
        run_id=build_run_id(
            task="eval_attack", model=cfg.model.name, dataset=cfg.dataset.name,
            attack=attack_cfg.method, steps=attack_cfg.steps, epsilon=epsilon, seed=seed,
        ),
        cfg=cfg,
        seed=seed,
        device=device,
        model=model,
        loaders={"test": get_test_loader(cfg.dataset, batch_size=64)},
        logger=make_logger(cfg.paths.logs / "eval_attack.jsonl"),
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
    """
    Construct RunContext for robustness comparison between:
    - standard model
    - adversarially trained (defense) model

    Includes:
    - two models (baseline + defense)
    - evaluation attack function
    - test data loader
    """
    device       = setup(cfg, seed)
    defense_tag  = attack_tag(training_cfg.attack)

    base_model    = load_model(
        str(cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"),
        device, cfg.model,
    )
    defense_model = load_model(
        str(cfg.paths.checkpoints / f"adv_{defense_tag}_seed{seed}" / "final.pth"),
        device, cfg.model,
    )

    eval_attack_fn, eval_attack_params = build_attack(
        with_epsilon(eval_cfg, epsilon)
    )

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
        logger=make_logger(cfg.paths.logs / "eval_robustness.jsonl"),
        attack_fn=eval_attack_fn,
        attack_params=eval_attack_params,
        epsilon=epsilon,
    )