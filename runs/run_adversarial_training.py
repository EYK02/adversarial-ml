# runs/run_adversarial_training.py

import sys
from pathlib import Path
from src.utils.runner import Experiment, ExperimentRunner

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

if args.dry_run:
    from src.utils.config import NUM_SEEDS_DRY as NUM_SEEDS, DEFENSES_DRY as DEFENSES
else:
    from src.utils.config import NUM_SEEDS, DEFENSES

TRAINING_EPSILON = 0.2

runner = ExperimentRunner()
experiments = []

for defense_attack, defense_steps in DEFENSES:
    for seed in range(NUM_SEEDS):
        if defense_attack == "pgd" and defense_steps is not None:
            attack_tag = f"pgd{defense_steps}"
        else:
            attack_tag = defense_attack

        model_path = Path(f"models/cnn_mnist_adv_{attack_tag}_eps{TRAINING_EPSILON}_seed{seed}.pth")

        if model_path.exists():
            print(f"Skipping {model_path.name} — already exists")
            continue

        cmd = [
            sys.executable, "-m", "src.training.adversarial_training",
            "--attack",  defense_attack,
            "--epsilon", str(TRAINING_EPSILON),
            "--seed",    str(seed),
        ]
        if defense_steps is not None:
            cmd.extend(["--steps", str(defense_steps)])

        attack_tag_label = f"{defense_attack}{defense_steps or ''}"
        experiments.append(Experiment(
            f"adv_train {attack_tag_label} seed={seed}",
            cmd,
        ))

for exp in experiments:
    runner.run(exp)

runner.summary()