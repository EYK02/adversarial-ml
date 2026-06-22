# src/runner/context_builders.py

from pathlib import Path

import torch

from src.attacks.registry import build_attack
from src.datasets.mnist import get_test_loader, get_train_loader
from src.models.factory import load_model, load_or_create_model
from src.runner.context import RunContext
from src.runner.utils import ckpt_paths, make_ckpt_dir, make_logger, resolve_alpha, setup, attack_tag, build_run_id
from src.utils.config import AttackConfig, ExperimentConfig, TrainingConfig


def build_train_ctx(
    cfg:          ExperimentConfig,
    training_cfg: TrainingConfig,
    seed:         int,
) -> RunContext:

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
    cfg:          ExperimentConfig,
    training_cfg: TrainingConfig,
    seed:         int,
) -> RunContext:

    device  = setup(cfg, seed)
    model   = load_or_create_model(cfg.model, device)
    tag     = attack_tag(training_cfg)
    epsilon = training_cfg.epsilon

    alpha = resolve_alpha(training_cfg.attack, epsilon)  # ← fixes the string-compare bug

    attack_cfg = AttackConfig(
        name=training_cfg.attack.name,
        epsilon=epsilon,
        steps=training_cfg.attack.steps,
        alpha=alpha,
        restarts=training_cfg.attack.restarts,
    )
    attack_fn, attack_params = build_attack(attack_cfg)

    ckpt_dir = make_ckpt_dir(cfg.paths.checkpoints, f"adv_{tag}_seed{seed}")
    latest, final, best = ckpt_paths(ckpt_dir)

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
        logger=make_logger(cfg.paths.logs / f"adv_{tag}_seed{seed}.jsonl"),
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

    device = setup(cfg, seed)
    model  = load_model(
        str(cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"),
        device, cfg.model,
    )

    alpha = resolve_alpha(attack_cfg, epsilon)

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

    device       = setup(cfg, seed)
    defense_tag  = attack_tag(training_cfg)

    base_model    = load_model(
        str(cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"),
        device, cfg.model,
    )
    defense_model = load_model(
        str(cfg.paths.checkpoints / f"adv_{defense_tag}_seed{seed}" / "final.pth"),
        device, cfg.model,
    )

    alpha = resolve_alpha(eval_cfg, epsilon)

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
        logger=make_logger(cfg.paths.logs / "eval_robustness.jsonl"),
        attack_fn=eval_attack_fn,
        attack_params=eval_attack_params,
        epsilon=epsilon,
    )