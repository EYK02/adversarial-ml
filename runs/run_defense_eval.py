# run_defense_eval.py

import sys
from src.utils.runner import Experiment, ExperimentRunner
from src.utils.config import EVAL_ATTACKS, DEFENSES, NUM_SEEDS

DEFENSE_EPSILON = 0.2

runner = ExperimentRunner()
experiments = []

for defense_attack, defense_steps in DEFENSES:
    for seed in range(NUM_SEEDS):
        for eval_attack, eval_steps in EVAL_ATTACKS:

            cmd = [
                sys.executable, "-m", "src.evaluation.eval_robustness",
                "--defense_attack",  defense_attack,
                "--defense_epsilon", str(DEFENSE_EPSILON),
                "--eval_attack",     eval_attack,
                "--seed",            str(seed),
            ]

            if defense_steps is not None:
                cmd.extend(["--defense_steps", str(defense_steps)])

            if eval_steps is not None:
                cmd.extend(["--eval_steps", str(eval_steps)])

            defense_tag = f"{defense_attack}{defense_steps or ''}"
            eval_tag    = f"{eval_attack}{eval_steps or ''}"

            experiments.append(
                Experiment(
                    f"def={defense_tag} atk={eval_tag} seed={seed}",
                    cmd,
                )
            )

for exp in experiments:
    runner.run(exp)

runner.summary()