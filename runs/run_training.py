# run_training.py

import sys
from src.utils.runner import Experiment, ExperimentRunner

runner = ExperimentRunner()

experiments = []

for seed in range(5):
    experiments.append(
        Experiment(
            f"Train seed {seed}",
            [
                sys.executable,
                "train.py",
                "--seed",
                str(seed)
            ]
        )
    )

for exp in experiments:
    runner.run(exp)