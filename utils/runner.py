# utils/runner.py

import subprocess
from dataclasses import dataclass

@dataclass
class Experiment:
    name: str
    command: list[str]

class ExperimentRunner:

    def run(self, experiment):
        print(f"\n=== {experiment.name} ===")

        try:
            subprocess.run(
                experiment.command,
                check=True
            )
            print("SUCCESS")

        except subprocess.CalledProcessError as e:
            print(f"FAILED ({e.returncode})")