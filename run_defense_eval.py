# run_defense_eval.py

import sys
from utils.runner import Experiment, ExperimentRunner

runner = ExperimentRunner()

experiments = []

for seed in range(5):
    experiments.append(
        Experiment(
            f"FGSM defense seed {seed}",
            [
                sys.executable,
                "-m",
                "defenses.evaluate_defense",
                "--attack",
                "fgsm",
                "--defense",
                "fgsm",
                "--seed",
                str(seed)
            ]
        )
    )

for exp in experiments:
    runner.run(exp)