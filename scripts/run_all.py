# scripts/run_all.py

import sys
import argparse
from pathlib import Path

from src.utils.config import load_experiment, ExperimentConfig, TrainingConfig
from src.utils.runner import Experiment, ExperimentRunner


def _attack_tag(training_cfg: TrainingConfig) -> str:
    attack = training_cfg.attack
    if attack.steps is not None:
        return f"{attack.name}{attack.steps}"
    return attack.name


def build_experiments(cfg: ExperimentConfig, dry_run: bool) -> list[tuple[str, list[Experiment]]]:
    """
    Returns an ordered list of (stage_name, [Experiment, ...]) tuples.
    Each stage is run sequentially; experiments within a stage are
    run sequentially by the runner.
    """
    dry_flag  = ["--dry-run"] if dry_run else []
    exp_flag = ["--experiment", cfg.experiment_path]
    py        = sys.executable

    stages = []

    # ── Stage 1: Standard training ───────────────────────────────
    stage1 = [
        Experiment(
            f"standard train seed={seed}",
            [py, "-m", "src.training.standard",
             "--experiment", exp_flag[1],
             "--seed", str(seed)]
            + dry_flag,
        )
        for seed in cfg.seeds
    ]
    stages.append(("STAGE 1 — Standard training", stage1))

    # ── Stage 2: Attack evaluation (baseline model) ───────────────
    stage2 = [
        Experiment(
            f"attack eval {a.name}{a.steps or ''} seed={seed}",
            [py, "-m", "src.evaluation.eval_attack",
             "--experiment", exp_flag[1],
             "--attack", f"{a.name}{a.steps or ''}",
             "--seed", str(seed)]
            + dry_flag,
        )
        for seed in cfg.seeds
        for a in cfg.eval_attacks
    ]
    stages.append(("STAGE 2 — Attack evaluation", stage2))

    # ── Stage 3: Adversarial training ────────────────────────────
    stage3 = [
        Experiment(
            f"adv train {_attack_tag(t)} seed={seed}",
            [py, "-m", "src.training.adversarial",
             "--experiment", exp_flag[1],
             "--training-config", _attack_tag(t),
             "--seed", str(seed)]
            + dry_flag,
        )
        for t in cfg.training
        if t.method == "adversarial"
        for seed in cfg.seeds
    ]
    stages.append(("STAGE 3 — Adversarial training", stage3))

    # ── Stage 4: Defense evaluation ───────────────────────────────
    stage4 = [
        Experiment(
            f"defense eval def={_attack_tag(t)} eval={a.name}{a.steps or ''} seed={seed}",
            [py, "-m", "src.evaluation.eval_robustness",
             "--experiment",      exp_flag[1],
             "--training-config", _attack_tag(t),
             "--eval-attack",     f"{a.name}{a.steps or ''}",
             "--seed",            str(seed)]
            + dry_flag,
        )
        for t in cfg.training
        if t.method == "adversarial"
        for a in cfg.eval_attacks
        for seed in cfg.seeds
    ]
    stages.append(("STAGE 4 — Defense evaluation", stage4))

    # ── Stage 5: Analysis report ──────────────────────────────────
    stage5 = [
        Experiment(
            "analysis report",
            [py, "-m", "src.analysis.report",
             "--experiment", exp_flag[1]]
            + dry_flag,
        )
    ]
    stages.append(("STAGE 5 — Analysis report", stage5))

    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment config, e.g. configs/experiments/mnist_cross_eval.yaml")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--stage",      type=int, default=None,
                        help="Run a single stage only (1-5)")
    args = parser.parse_args()

    cfg    = load_experiment(args.experiment, dry_run=args.dry_run)
    stages = build_experiments(cfg, dry_run=args.dry_run)

    # create run directories
    cfg.paths.logs.mkdir(parents=True,        exist_ok=True)
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)
    cfg.paths.metrics.mkdir(parents=True,     exist_ok=True)
    cfg.paths.figures.mkdir(parents=True,     exist_ok=True)

    runner = ExperimentRunner()

    for i, (stage_name, experiments) in enumerate(stages, start=1):
        if args.stage is not None and args.stage != i:
            continue

        print(f"\n{'═' * 60}")
        print(f"  {stage_name}  ({len(experiments)} jobs)")
        print(f"{'═' * 60}")

        for exp in experiments:
            runner.run(exp)

    runner.summary()


if __name__ == "__main__":
    main()