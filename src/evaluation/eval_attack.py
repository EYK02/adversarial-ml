# src/evaluation/eval_attack.py

import argparse
import time

import sys

from src.evaluation.core import evaluate
from src.utils.config import load_experiment
from src.utils.context import build_eval_attack_ctx, RunContext


def eval_attack(ctx: RunContext) -> None:
    if ctx.logger.contains(ctx.run_id):
        print(f"  [SKIP] eps={ctx.epsilon:.2f}")
        return False

    start = time.perf_counter()

    acc = evaluate(ctx)

    duration = time.perf_counter() - start

    print(
        f"eps={ctx.epsilon:.2f} | acc={acc:.2f}% | time={duration:.1f}s"
    )

    ctx.logger.log({
        "run_id": ctx.run_id,
        "accuracy": float(acc),
        "epsilon": float(ctx.epsilon),
        "duration_sec": float(duration),
    })

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_experiment(
        args.experiment,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        run_name=args.run_name
    )

    attack_cfg = next(
        a for a in cfg.eval_attacks
        if (a.name == args.attack)
        or (a.steps is not None and f"{a.name}{a.steps}" == args.attack)
    )

    print(f"Attack eval — {args.attack}, seed={args.seed}")

    ran_any = False

    for epsilon in cfg.epsilon_eval:
        ctx = build_eval_attack_ctx(
            cfg=cfg,
            attack_cfg=attack_cfg,
            seed=args.seed,
            epsilon=epsilon,
        )

        if eval_attack(ctx):
            ran_any = True
    
    if not ran_any:
        sys.exit(2)


if __name__ == "__main__":
    main()