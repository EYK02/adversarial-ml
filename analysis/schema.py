# analysis/schema.py

import pandas as pd

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
    steps:        Optional[int]   = None
    alpha:        Optional[float] = None


@dataclass
class AdvTrainingRecord:
    run_type:             RunType
    model:                str
    dataset:              str
    seed:                 int
    attack:               str
    epsilon:              float
    epoch:                int
    train_loss:           float
    train_clean_accuracy: float
    train_adv_accuracy:   float
    test_clean_accuracy:  float
    timestamp:            str
    steps:                Optional[int] = None


@dataclass
class DefenseEvalRecord:
    run_type:            RunType
    model:               str
    dataset:             str
    seed:                int
    defense_attack:      str
    defense_epsilon:     float
    defense_path:        str
    eval_attack:         str
    epsilon:             float
    baseline_accuracy:   float
    defense_accuracy:    float
    delta:               float
    duration_sec:        float
    timestamp:           str
    defense_steps:       Optional[int] = None
    eval_steps:          Optional[int] = None


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
        accuracy     = float(row["value"]),
        duration_sec = float(row["duration_sec"]),
        seed         = int(row["seed"]),
        timestamp    = row["timestamp"],
        steps        = int(params["steps"]) if "steps" in params else None,
        alpha        = float(params["alpha"]) if "alpha" in params else None,
    ))


def _normalize_adv_training(row: dict[str, Any]) -> dict[str, Any]:
    raw_params = row.get("attack_params")
    params = raw_params if isinstance(raw_params, dict) else {}

    return asdict(AdvTrainingRecord(
        run_type             = row["run_type"],
        model                = row["model"],
        dataset              = row["dataset"],
        seed                 = int(row["seed"]),
        attack               = row["attack"],
        epsilon              = float(row["epsilon"]),
        epoch                = int(row["epoch"]),
        train_loss           = float(row["train_loss"]),
        train_clean_accuracy = float(row["train_clean_accuracy"]),
        train_adv_accuracy   = float(row["train_adv_accuracy"]),
        test_clean_accuracy  = float(row["test_clean_accuracy"]),
        timestamp            = row["timestamp"],
        steps                = int(params["steps"]) if "steps" in params else None,
    ))


def _normalize_defense_eval(row: dict[str, Any]) -> dict[str, Any]:
    defense_params = row.get("defense_params") or {}
    eval_params    = row.get("eval_params")    or {}

    return asdict(DefenseEvalRecord(
        run_type          = row["run_type"],
        model             = row["model"],
        dataset           = row["dataset"],
        seed              = int(row["seed"]),
        defense_attack    = row["defense_attack"],
        defense_epsilon   = float(row["defense_epsilon"]),
        defense_path      = row["defense_path"],
        eval_attack       = row["eval_attack"],
        epsilon           = float(row["epsilon"]),
        baseline_accuracy = float(row["baseline_accuracy"]),
        defense_accuracy  = float(row["defense_accuracy"]),
        delta             = float(row["delta"]),
        duration_sec      = float(row["duration_sec"]),
        timestamp         = row["timestamp"],
        defense_steps     = int(defense_params["steps"]) if "steps" in defense_params else None,
        eval_steps        = int(eval_params["steps"])    if "steps" in eval_params    else None,
    ))


# ─────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────

_NORMALIZERS = {
    "training":    _normalize_training,
    "attack_eval": _normalize_attack_eval,
    "adv_training": _normalize_adv_training,
    "defense_eval": _normalize_defense_eval,
}


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    run_type = row.get("run_type")
    fn = _NORMALIZERS.get(run_type)

    if fn is None:
        raise ValueError(f"No normalizer for run_type: {run_type!r}")

    return fn(row)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    rows = [normalize_row(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows)