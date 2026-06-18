# runs/run_training.py

import sys
from src.utils.runner import Experiment, ExperimentRunner


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

if args.dry_run:
    from src.utils.config import NUM_SEEDS_DRY as NUM_SEEDS
else:
    from src.utils.config import NUM_SEEDS

dry_flag = ["--dry-run"] if args.dry_run else []


runner = ExperimentRunner()

experiments = []

for seed in range(NUM_SEEDS):
    experiments.append(
        Experiment(
            f"Train seed {seed}",
            [
                sys.executable, "-m", "src.training.train",
                "--seed", str(seed)
            ] + dry_flag,
        )
    )

for exp in experiments:
    runner.run(exp)

runner.summary()