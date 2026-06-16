# analysis/schema.py

from dataclasses import dataclass, asdict
from typing import Any, Optional, Literal

RunType = Literal["training", "attack_eval", "adv_training", "defense_eval"]


# ─────────────────────────────────────────
# DATACLASSES  (authoritative schema docs)
# ─────────────────────────────────────────

@dataclass
class TrainingRecord:
    run_type:       RunType
    dataset:        str
    model:          str
    seed:           int
    epoch:          int
    train_loss:     float
    train_accuracy: float
    test_loss:      float
    test_accuracy:  float
    timestamp:      str


@dataclass
class AttackEvalRecord:
    run_type:     RunType
    model:        str
    model_path:   str
    dataset:      str
    attack:       str
    epsilon:      float
    accuracy:     float
    duration_sec: float
    seed:         int
    timestamp:    str
    steps:        Optional[int]  = None
    alpha:        Optional[float] = None


# ─────────────────────────────────────────
# NORMALIZERS
# ─────────────────────────────────────────

def _normalize_training(row: dict[str, Any]) -> dict[str, Any]:
    return asdict(TrainingRecord(
        run_type       = row["run_type"],
        dataset        = row["dataset"],
        model          = row["model"],
        seed           = int(row["seed"]),
        epoch          = int(row["epoch"]),
        train_loss     = float(row["train_loss"]),
        train_accuracy = float(row["train_accuracy"]),
        test_loss      = float(row["test_loss"]),
        test_accuracy  = float(row["test_accuracy"]),
        timestamp      = row["timestamp"],
    ))


def _normalize_attack_eval(row: dict[str, Any]) -> dict[str, Any]:
    raw_params = row.get("attack_params")
    params = raw_params if isinstance(raw_params, dict) else {}

    return asdict(AttackEvalRecord(
        run_type     = row["run_type"],
        model        = row["model"],
        model_path   = row.get("model_path", ""),
        dataset      = row["dataset"],
        attack       = row["attack"],
        epsilon      = float(row["epsilon"]),
        accuracy     = float(row["value"]),       # "metric"/"value" -> "accuracy"
        duration_sec = float(row["duration_sec"]),
        seed         = int(row["seed"]),
        timestamp    = row["timestamp"],
        steps        = int(params["steps"]) if "steps" in params else None,
        alpha        = float(params["alpha"]) if "alpha" in params else None,
    ))


# ─────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────

_NORMALIZERS = {
    "training":    _normalize_training,
    "attack_eval": _normalize_attack_eval,
}

def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    run_type = row.get("run_type")
    fn = _NORMALIZERS.get(run_type)

    if fn is None:
        raise ValueError(f"No normalizer for run_type: {run_type!r}")

    return fn(row)