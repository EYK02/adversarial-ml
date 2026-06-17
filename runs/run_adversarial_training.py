# runs/run_adversarial_training.py

import sys
from pathlib import Path
from src.utils.runner import Experiment, ExperimentRunner

TRAINING_EPSILON = 0.2
SEEDS = list(range(5))

DEFENSES = [
    ("fgsm", None),
    ("pgd",  5),
    ("pgd",  10),
    ("pgd",  20),
    ("pgd",  40),
]

runner = ExperimentRunner()
experiments = []

for defense_attack, defense_steps in DEFENSES:
    for seed in SEEDS:
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