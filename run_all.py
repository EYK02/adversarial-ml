# run_all.py

import sys
import argparse
from src.utils.runner import Experiment, ExperimentRunner

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

dry_flag = ["--dry-run"] if args.dry_run else []

runner = ExperimentRunner()

stages = [
    ("STAGE 1 — Baseline training",    "runs.run_training"),
    ("STAGE 2 — Attack evaluation",    "runs.run_attack_eval"),
    ("STAGE 3 — Adversarial training", "runs.run_adversarial_training"),
    ("STAGE 4 — Defense evaluation",   "runs.run_defense_eval"),
    ("STAGE 5 — Analysis report",      "analysis.report"),
]

for stage_name, module in stages:
    print(f"\n{'═' * 60}")
    print(f"  {stage_name}")
    print(f"{'═' * 60}")
    runner.run(Experiment(stage_name, [sys.executable, "-m", module] + dry_flag))

runner.summary()