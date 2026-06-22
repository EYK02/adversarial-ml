# src/evaluation/eval_robustness.py

import argparse
import time
import sys

from src.evaluation.utils import evaluate
from src.runner.builders import build_eval_robustness_ctx
from src.runner.utils import attack_tag
from src.utils.config import load_experiment
from src.runner.context import RunContext


def eval_robustness(ctx: RunContext) -> None:
    if ctx.logger.contains(ctx.run_id):
        print(f"    [SKIP] eps={ctx.epsilon:.2f}")
        return False
    
    start = time.perf_counter()

    base_acc = evaluate(ctx)
    defense_acc = evaluate(ctx)

    duration = time.perf_counter() - start

    print(
        f"eps={ctx.epsilon:.2f} | "
        f"base={base_acc:.2f}% | "
        f"defense={defense_acc:.2f}% | "
        f"delta={defense_acc - base_acc:+.2f}% | "
        f"time={duration:.1f}s"
    )

    ctx.logger.log({
        "run_id": ctx.run_id,
        "run_type": "eval_robustness",
        "model": ctx.cfg.model.name,
        "dataset": ctx.cfg.dataset.name,
        "seed": ctx.seed,

        "defense_attack": ctx.training_cfg.attack.name,
        "defense_params": {
            "steps": ctx.training_cfg.attack.steps
        } if ctx.training_cfg.attack.steps else {},
        "defense_epsilon": float(ctx.training_cfg.epsilon),

        "eval_attack": ctx.attack_params,
        "epsilon": float(ctx.epsilon),

        "baseline_accuracy": float(base_acc),
        "defense_accuracy": float(defense_acc),
        "delta": float(defense_acc - base_acc),
        "duration_sec": float(duration),
    })

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--training-config", type=str, required=True)
    parser.add_argument("--eval-attack", type=str, required=True)
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

    training_cfg = next(
        t for t in cfg.training
        if t.method == "adversarial"
        and attack_tag(t) == args.training_config
    )

    eval_cfg = next(
        a for a in cfg.eval_attacks
        if (a.name == args.eval_attack)
        or (a.steps is not None and f"{a.name}{a.steps}" == args.eval_attack)
    )

    print(
        f"Defense eval — defense={args.training_config}, "
        f"eval={args.eval_attack}, seed={args.seed}"
    )

    for epsilon in cfg.epsilon_eval:

        ctx = build_eval_robustness_ctx(
            cfg=cfg,
            training_cfg=training_cfg,
            eval_cfg=eval_cfg,
            seed=args.seed,
            epsilon=epsilon,
        )

        ran_any = False
        for epsilon in cfg.epsilon_eval:
            ctx = build_eval_robustness_ctx(...)
            if eval_robustness(ctx):
                ran_any = True

        if not ran_any:
            sys.exit(2)


if __name__ == "__main__":
    main()