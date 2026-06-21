# src/evaluation/eval_attack.py

import argparse
import time

from src.attacks.registry import build_attack
from src.datasets.mnist import get_test_loader
from src.evaluation.core import evaluate
from src.models.factory import load_model
from src.utils.config import load_experiment, ExperimentConfig, AttackConfig
from src.utils.context import build_attack_ctx, RunContext
from src.utils.logger import JSONLLogger
from src.utils.run_id import make_run_id
from src.utils.seed import set_seed, get_device


def eval_attack(ctx: RunContext) -> None:

    device = ctx.device
    model = ctx.model
    test_loader = ctx.loaders["test"]
    epsilon = ctx.epsilon

    run_id = make_run_id(
        task="eval_attack",
        model=ctx.cfg.model.name,
        dataset=ctx.cfg.dataset.name,
        attack=ctx.attack_params["name"] if ctx.attack_params else "unknown",
        steps=ctx.attack_params.get("steps") if ctx.attack_params else None,
        epsilon=epsilon,
        seed=ctx.seed,
    )

    if ctx.logger.contains(run_id):
        print(f"  [SKIP] {run_id} already completed.")
        return

    start = time.perf_counter()

    acc = evaluate(
        ctx=ctx,
        attack_fn=ctx.attack_fn,
        split="test",
        epsilon=epsilon,
    )

    duration = time.perf_counter() - start

    print(
        f"  eps={epsilon:.2f} | "
        f"acc={acc:.2f}% | "
        f"duration={duration:.1f}s"
    )

    ctx.logger.log({
        "run_id": run_id,
        "run_type": "eval_attack",
        "model": ctx.cfg.model.name,
        "model_path": str(ctx.cfg.paths.checkpoints),
        "dataset": ctx.cfg.dataset.name,
        "seed": ctx.seed,
        "attack": ctx.attack_params,
        "epsilon": float(epsilon),
        "accuracy": float(acc),
        "duration_sec": float(duration),
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_experiment(
        args.config,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        run_name=args.run_name
    )

    attack_cfg = next(
        a for a in cfg.eval_attacks
        if (a.name == args.attack)
        or (a.steps is not None and f"{a.name}{a.steps}" == args.attack)
    )

    log_path = cfg.paths.logs / "eval_attack.jsonl"
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(str(log_path))

    set_seed(args.seed)
    print(f"Attack eval — {args.attack}, seed={args.seed}")

    # ---- load model ONCE ----
    device = get_device()

    checkpoint_path = cfg.paths.checkpoints / f"standard_seed{args.seed}" / "final.pth"
    model = load_model(str(checkpoint_path), device, cfg.model)

    # ---- resolve attack ONCE ----
    steps = attack_cfg.steps
    alpha = attack_cfg.alpha
    if alpha is None and steps is not None:
        alpha = 2.5 * cfg.epsilon_eval[0] / steps  # placeholder; overwritten per eps anyway

    attack_cfg_resolved = AttackConfig(
        name=attack_cfg.name,
        steps=steps,
        alpha=alpha,
        restarts=attack_cfg.restarts,
    )

    attack_fn, attack_params = build_attack(attack_cfg_resolved)

    # ---- loaders ONCE ----
    test_loader = get_test_loader(cfg.dataset, batch_size=64)

    # ---- evaluation loop ----
    for epsilon in cfg.epsilon_eval:

        ctx = build_attack_ctx(
            cfg=cfg,
            attack_cfg=attack_cfg_resolved,
            seed=args.seed,
            epsilon=epsilon,
            device=device,
            model=model,
            test_loader=test_loader,
            attack_fn=attack_fn,
            attack_params=attack_params,
            logger=logger,
        )

        eval_attack(ctx)


if __name__ == "__main__":
    main()