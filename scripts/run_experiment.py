# scripts/run_experiment.py

import sys
import argparse
import shutil
import time
from pathlib import Path

from src.utils.config import load_experiment, ExperimentConfig, TrainingConfig
from src.utils.runner import Experiment, ExperimentRunner


def _attack_tag(training_cfg: TrainingConfig) -> str:
    attack = training_cfg.attack
    if attack.steps is not None:
        return f"{attack.name}{attack.steps}"
    return attack.name


def build_experiments(
        cfg:        ExperimentConfig, 
        dry_run:    bool, 
        smoke_test: bool,
        run_name:   str,
    ) -> list[tuple[str, list[Experiment]]]:
    """
    Returns an ordered list of (stage_name, [Experiment, ...]) tuples.
    Each stage is run sequentially; experiments within a stage are
    run sequentially by the runner.
    """
    dry_flag   = ["--dry-run"]    if dry_run    else []
    smoke_flag = ["--smoke-test"] if smoke_test else []
    name_flag  = ["--run-name", run_name] if run_name else []
    mode_flag = dry_flag + smoke_flag + name_flag

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
            + mode_flag,
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
            + mode_flag,
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
            + mode_flag,
        )
        for t in cfg.training
        if t.method == "adversarial"
        for seed in cfg.seeds
    ]
    stages.append(("STAGE 3 — Adversarial training", stage3))

    # ── Stage 4: Robustness evaluation ───────────────────────────────
    stage4 = [
        Experiment(
            f"robustness eval def={_attack_tag(t)} eval={a.name}{a.steps or ''} seed={seed}",
            [py, "-m", "src.evaluation.eval_robustness",
             "--experiment",      exp_flag[1],
             "--training-config", _attack_tag(t),
             "--eval-attack",     f"{a.name}{a.steps or ''}",
             "--seed",            str(seed)]
            + mode_flag,
        )
        for t in cfg.training
        if t.method == "adversarial"
        for a in cfg.eval_attacks
        for seed in cfg.seeds
    ]
    stages.append(("STAGE 4 — Robustness evaluation", stage4))

    # ── Stage 5: Analysis report ──────────────────────────────────
    stage5 = [
        Experiment(
            "analysis report",
            [py, "-m", "src.analysis.report",
             "--experiment", exp_flag[1]]
            + mode_flag,
        )
    ]
    stages.append(("STAGE 5 — Analysis report", stage5))

    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config, e.g. configs/experiments/mnist_cross_eval.yaml")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--smoke-test",    action="store_true")
    parser.add_argument("--stage",      type=int, default=None,
                        help="Run a single stage only (1-5)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override run name for resuming previous run, e.g. 20226-06-21_mnist_cross_eval")
    args = parser.parse_args()

    start_time = time.perf_counter()

    cfg    = load_experiment(
        args.config, 
        dry_run=args.dry_run, 
        smoke_test=args.smoke_test, 
        run_name=args.run_name
    )
    stages = build_experiments(
        cfg, 
        dry_run=args.dry_run, 
        smoke_test=args.smoke_test,
        run_name=args.run_name
    )

    # create run directories
    for p in [
        cfg.paths.logs, 
        cfg.paths.metrics, 
        cfg.paths.figures
        ]:
        p.mkdir(parents=True, exist_ok=True)

    snapshot_path = cfg.paths.run_dir / "config.yaml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    if not snapshot_path.exists():
        shutil.copy2(cfg.experiment_path, snapshot_path)

    runner = ExperimentRunner()

    # count total jobs for ETA
    all_jobs = [
        exp
        for i, (_, experiments) in enumerate(stages, start=1)
        if args.stage is None or args.stage == i
        for exp in experiments
    ]
    runner.set_total(len(all_jobs))

    print(f"\n{'═' * 60}")
    print(f"  Run : {cfg.run_name}")
    print(f"  Config : {cfg.experiment_path}")
    print(f"  Dry run : {cfg.dry_run}")
    print(f"  Smoke test : {cfg.smoke_test}")
    print(f"  Seeds : {cfg.seeds}")
    print(f"  Jobs : {len(all_jobs)}")
    print(f"{'═' * 60}")

    for i, (stage_name, experiments) in enumerate(stages, start=1):
        if args.stage is not None and args.stage != i:
            continue

        print(f"\n{'═' * 60}")
        print(f"  {stage_name}  ({len(experiments)} jobs)")
        print(f"{'═' * 60}")

        for exp in experiments:
            runner.run(exp)

    runner.summary()

    duration = time.perf_counter() - start_time
    print(f"Total elapsed time: {duration:.1f}s")


if __name__ == "__main__":
    main()