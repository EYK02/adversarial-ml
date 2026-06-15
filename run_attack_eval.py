
from utils.runner import Experiment, ExperimentRunner

runner = ExperimentRunner()

experiments = []

for seed in range(5):
    experiments.append(
        Experiment(
            f"FGSM seed {seed}",
            [
                "python",
                "-m",
                "attacks.evaluate_attack",
                "--attack",
                "fgsm",
                "--seed",
                str(seed)
            ]
        )
    )

for exp in experiments:
    runner.run(exp)