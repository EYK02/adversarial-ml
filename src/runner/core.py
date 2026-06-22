from typing import Optional

from src.utils.config import ExperimentConfig
from src.runner.executor import ExperimentRunner, Experiment
from src.runner.builders import build_experiments


def run_single_experiment(runner: ExperimentRunner, exp: Experiment) -> None:
    """
    Execute one experiment using the shared runner.
    """
    runner.run(exp)


def run_stages(
    cfg: ExperimentConfig,
    runner: Optional[ExperimentRunner] = None,
    stage: Optional[int] = None,
    dry_run: bool = False,
    smoke_test: bool = False,
    run_name: str = None,
) -> ExperimentRunner:
    
    if runner is None:
        runner = ExperimentRunner()

    stages = build_experiments(
        cfg=cfg,
        dry_run=dry_run,
        smoke_test=smoke_test,
        run_name=run_name,
    )

    # flatten experiments for ETA
    all_jobs = [
        exp
        for i, s in enumerate(stages, start=1)
        if stage is None or stage == i
        for exp in s.experiments
    ]

    runner.set_total(len(all_jobs))

    # execute stages
    for i, s in enumerate(stages, start=1):

        if stage is not None and stage != i:
            continue

        print(f"\n{'═' * 60}")
        print(f"  {s.name} ({len(s.experiments)} jobs)")
        print(f"{'═' * 60}")

        for exp in s.experiments:
            runner.run(exp)

    runner.summary()
    return runner