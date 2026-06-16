# run_adversarial_training.py

import sys
from src.utils.runner import Experiment, ExperimentRunner

TRAINING_EPSILON = 0.2
SEEDS = list(range(5))

PGD_STEPS = [5, 10, 20, 40]

runner = ExperimentRunner()

experiments = []

for seed in SEEDS:
    # FGSM defense
    experiments.append(
        Experiment(
            f"FGSM defense seed={seed}",
            [
                sys.executable, "-m", "src.training.adversarial_training",
                "--attack",  "fgsm",
                "--epsilon", str(TRAINING_EPSILON),
                "--seed",    str(seed),
            ]
        )
    )

# PGD defenses
for steps in PGD_STEPS:
    for seed in SEEDS:      
        experiments.append(
            Experiment(
                f"PGD-{steps} defense seed={seed}",
                [
                    sys.executable, "-m", "src.training.adversarial_training",
                    "--attack",  "pgd",
                    "--steps",   str(steps),
                    "--epsilon", str(TRAINING_EPSILON),
                    "--seed",    str(seed),
                ]                                           
            )
        )

for exp in experiments:
    runner.run(exp)

runner.summary()