# src/utils/runner.py

import subprocess
from dataclasses import dataclass

@dataclass
class Experiment:
    name: str
    command: list[str]

class ExperimentRunner:
    def __init__(self):
        self.failed = []

    def run(self, experiment):
        print(f"\n=== {experiment.name} ===")
        try:
            subprocess.run(experiment.command, check=True)
            print("SUCCESS")
        except subprocess.CalledProcessError as e:
            print(f"FAILED ({e.returncode})")
            self.failed.append(experiment.name)

    def summary(self):
        if self.failed:
            print(f"\n{len(self.failed)} experiments failed:")
            for name in self.failed:
                print(f"  - {name}")
        else:
            print("\nAll experiments succeeded.")