# src/runner/runner.py

"""
Experiment execution engine.

Provides a lightweight orchestration layer for running experiments
as subprocess-based jobs in a staged pipeline.

Responsibilities:
- Execute experiments sequentially
- Track progress, timing, and failures
- Estimate remaining runtime (ETA)
- Provide summary reports
"""

import subprocess
import time
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class Experiment:
    """
    Single executable experiment.

    Attributes
    ----------
    name:
        Human-readable experiment name.

    command:
        Full subprocess command to execute.
    """
    name:    str
    command: list[str]


@dataclass
class ExperimentRunner:
    failed:    list[str] = field(default_factory=list)
    completed: int       = 0
    total:     int       = 0
    _times:    list[float] = field(default_factory=list)
    _start:    float       = field(default_factory=time.perf_counter)
    """
    Execution manager for a sequence of experiments.

    Handles:
    - sequential execution of subprocess jobs
    - failure tracking
    - timing statistics
    - ETA estimation
    - final summary reporting
    """

    def set_total(self, n: int) -> None:
        self.total  = n
        self._start = time.perf_counter()

    def run(self, experiment: Experiment) -> None:
        """
        Execute a single experiment subprocess.

        Tracks execution time, handles failures, and updates
        internal progress statistics.
        """
        print(f"\n=== {experiment.name} ===")
        job_start = time.perf_counter()
        skipped = False

        try:
            subprocess.run(experiment.command, check=True)
            # print("SUCCESS")
        except subprocess.CalledProcessError as e:
            if e.returncode == 2:
                print("[SKIP] job already completed")
                skipped = True
            else:
                print(f"FAILED ({e.returncode})")
                self.failed.append(experiment.name)

        duration = time.perf_counter() - job_start
        if not skipped:
            self._times.append(duration)
        self.completed += 1

        self._print_eta()

    def _print_eta(self) -> None:
        if self.total == 0 or not self._times:
            return

        avg        = sum(self._times) / len(self._times)
        remaining  = self.total - self.completed
        eta_secs   = avg * remaining
        elapsed    = time.perf_counter() - self._start

        print(
            f"  Progress : {self.completed}/{self.total} jobs"
            f"  |  elapsed {_fmt(elapsed)}"
            f"  |  avg/job {_fmt(avg)}"
            f"  |  ETA {_fmt(eta_secs)}"
        )

    def summary(self) -> None:
        """
        Print final execution summary including:
        - total runtime
        - number of completed experiments
        - failed experiments (if any)
        """
        elapsed = time.perf_counter() - self._start
        print(f"\n{'═' * 60}")
        print(f"  Finished {self.completed} jobs in {_fmt(elapsed)}")
        if self.failed:
            print(f"  {len(self.failed)} failed:")
            for name in self.failed:
                print(f"    - {name}")
        else:
            print("  All experiments succeeded.")
        print(f"{'═' * 60}")


def _fmt(seconds: float) -> str:
    """Format seconds as h:mm:ss or m:ss."""
    td  = timedelta(seconds=int(seconds))
    h   = td.seconds // 3600
    m   = (td.seconds % 3600) // 60
    s   = td.seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"
