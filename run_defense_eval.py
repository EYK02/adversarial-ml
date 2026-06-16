# run_defense_eval.py

import sys

from utils.runner import Experiment, ExperimentRunner

runner = ExperimentRunner()

experiments = []

seeds = range(1)

defenses = [
    ("fgsm", None),
    ("pgd", 10),
    ("pgd", 20),
    ("pgd", 5),
    ("pgd", 40),
]

eval_attacks = [
    ("pgd", 40),
]

for seed in seeds:
    for defense_attack, defense_steps in defenses:
        for eval_attack, eval_steps in eval_attacks:

            cmd = [
                sys.executable,
                "-m",
                "defenses.evaluate_defense",

                "--defense_attack",
                defense_attack,

                "--defense_epsilon",
                "0.2",

                "--eval_attack",
                eval_attack,

                "--seed",
                str(seed),
            ]

            if defense_steps is not None:
                cmd.extend([
                    "--defense_steps",
                    str(defense_steps),
                ])

            if eval_steps is not None:
                cmd.extend([
                    "--eval_steps",
                    str(eval_steps),
                ])

            experiments.append(
                Experiment(
                    (
                        f"def={defense_attack}"
                        f"{defense_steps or ''} "
                        f"atk={eval_attack}"
                        f"{eval_steps or ''} "
                        f"seed={seed}"
                    ),
                    cmd,
                )
            )

for exp in experiments:
    runner.run(exp)
