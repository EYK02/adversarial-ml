# run_attack_eval.py

import sys
from src.utils.runner import Experiment, ExperimentRunner
from src.utils.config import NUM_SEEDS

runner = ExperimentRunner()

pgd_steps = [10]

experiments = []

for seed in range(NUM_SEEDS):
    for steps in pgd_steps:
        experiments.append(
            Experiment(
                f"pgd steps={steps} seed={seed}",
                [
                    sys.executable, "-m", "src.evaluation.eval_attack",
                    "--attack", "pgd",
                    "--steps", str(steps),
                    "--seed", str(seed),
                ],
            )
        )

for exp in experiments:
    runner.run(exp)

runner.summary()