# runs/run_training.py

import sys
from src.utils.runner import Experiment, ExperimentRunner
from src.utils.config import NUM_SEEDS
runner = ExperimentRunner()

experiments = []

for seed in range(NUM_SEEDS):
    experiments.append(
        Experiment(
            f"Train seed {seed}",
            [
                sys.executable, "-m", "src.training.train",
                "--seed", str(seed)
            ]
        )
    )

for exp in experiments:
    runner.run(exp)

runner.summary()