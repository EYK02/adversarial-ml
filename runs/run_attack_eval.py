# run_attack_eval.py

import sys
from src.utils.runner import Experiment, ExperimentRunner

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

if args.dry_run:
    from src.utils.config import NUM_SEEDS_DRY as NUM_SEEDS, ATTACKS_DRY as ATTACKS 
else:
    from src.utils.config import NUM_SEEDS, ATTACKS as ATTACKS

dry_flag = ["--dry-run"] if args.dry_run else []

runner = ExperimentRunner()


experiments = []

for seed in range(NUM_SEEDS):
    for attack, steps in ATTACKS:
        cmd = [
            sys.executable, "-m", "src.evaluation.eval_attack",
            "--attack", attack,
            "--seed", str(seed),
        ]
        if steps is not None:
            cmd.extend(["--steps", str(steps)])
        cmd += dry_flag

        experiments.append(
            Experiment(f"{attack} steps={steps} seed={seed}", cmd)
        )

for exp in experiments:
    runner.run(exp)

runner.summary()