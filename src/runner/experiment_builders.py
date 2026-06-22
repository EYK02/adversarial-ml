# src/runner/experiment_builders.py


from dataclasses import dataclass
import sys

from src.runner.executor import Experiment
from src.runner.utils import attack_tag
from src.utils.config import ExperimentConfig


@dataclass
class Stage:
    name: str
    experiments: list[Experiment]


def build_experiments(
    cfg: ExperimentConfig,
    dry_run: bool = False,
    smoke_test: bool = False,
    run_name: str | None = None,
) -> list[Stage]:
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

    stages: list[Stage] = []

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
    stages.append(Stage("STAGE 1 — Standard training", stage1))

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
    stages.append(Stage("STAGE 2 — Attack evaluation", stage2))

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
    stages.append(Stage("STAGE 3 — Adversarial training", stage3))

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
    stages.append(Stage("STAGE 4 — Robustness evaluation", stage4))

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
    stages.append(Stage("STAGE 5 — Analysis report", stage5))

    return stages

