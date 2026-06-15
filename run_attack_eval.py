# run_attack_eval.py

import sys
from utils.runner import Experiment, ExperimentRunner

runner = ExperimentRunner()

experiments = []

for seed in range(5):
    experiments.append(
        Experiment(
            f"pgd5 seed {seed}",
            [
                sys.executable,
                "-m",
                "attacks.evaluate_attack",
                "--attack",
                "pgd5",
                "--seed",
                str(seed)
            ]
        )
    )

for exp in experiments:
    runner.run(exp)