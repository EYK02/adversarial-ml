# run_attack_eval.py

import sys
from src.utils.runner import Experiment, ExperimentRunner

runner = ExperimentRunner()

pgd_steps = [5, 10, 20, 40]
seeds = range(5)

experiments = []

for seed in seeds:
    for steps in pgd_steps:
        experiments.append(
            Experiment(
                f"pgd steps={steps} seed={seed}",
                [
                    sys.executable,
                    "-m",
                    "attacks.evaluate_attack",
                    "--attack",
                    "pgd",
                    "--steps",
                    str(steps),
                    "--seed",
                    str(seed),
                ],
            )
        )

for exp in experiments:
    runner.run(exp)