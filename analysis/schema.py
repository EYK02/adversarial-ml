# analysis/schema.py

from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal
import time


RunType = Literal[
    "training",
    "attack_eval",
    "adv_training",
    "defense_eval",
    "model_save"
]


# -------------------------
# BASE NORMALIZED RECORD
# -------------------------
@dataclass
class BaseRecord:
    run_type: RunType
    model: str
    dataset: str
    seed: int
    timestamp: float


# -------------------------
# ATTACK EVAL RECORD
# -------------------------
@dataclass
class AttackEvalRecord(BaseRecord):
    attack: str
    epsilon: float
    accuracy: float
    duration_sec: float

    # flattened attack params (IMPORTANT)
    steps: Optional[int] = None
    alpha: Optional[float] = None


# -------------------------
# TRAINING RECORD
# -------------------------
@dataclass
class TrainingRecord(BaseRecord):
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float


# -------------------------
# ADV TRAINING RECORD
# -------------------------
@dataclass
class AdvTrainingRecord(BaseRecord):
    attack: str
    epsilon: float
    epoch: int

    train_loss: float
    train_clean_accuracy: float
    train_adv_accuracy: float
    test_clean_accuracy: float

    steps: Optional[int] = None


# -------------------------
# MODEL SAVE RECORD
# -------------------------
@dataclass
class ModelSaveRecord(BaseRecord):
    model_path: str
    duration_sec: float

    attack: Optional[str] = None
    epsilon: Optional[float] = None


# -------------------------
# NORMALIZATION LOGIC
# -------------------------
def normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts raw JSONL row into strict flat schema-compatible dict.
    """

    base = {
        "run_type": row["run_type"],
        "model": row.get("model", "unknown"),
        "dataset": row.get("dataset", "mnist"),
        "seed": row.get("seed", -1),
        "timestamp": row.get("timestamp", time.time()),
    }

    # ---------------- attack eval ----------------
    if row["run_type"] == "attack_eval":
        params = row.get("attack_params", {}) or {}

        return {
            **base,
            "attack": row["attack"],
            "epsilon": row["epsilon"],
            "accuracy": row.get("value"),
            "duration_sec": row.get("duration_sec"),

            # flatten params
            "steps": params.get("steps"),
            "alpha": params.get("alpha"),
        }

    # ---------------- training ----------------
    if row["run_type"] == "training":
        return {
            **base,
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "train_accuracy": row["train_accuracy"],
            "test_loss": row["test_loss"],
            "test_accuracy": row["test_accuracy"],
        }

    # ---------------- adversarial training ----------------
    if row["run_type"] == "adv_training":
        return {
            **base,
            "attack": row["attack"],
            "epsilon": row["epsilon"],
            "epoch": row["epoch"],

            "train_loss": row["train_loss"],
            "train_clean_accuracy": row["train_clean_accuracy"],
            "train_adv_accuracy": row["train_adv_accuracy"],
            "test_clean_accuracy": row["test_clean_accuracy"],

            "steps": (row.get("attack_params") or {}).get("steps"),
        }

    # ---------------- model save ----------------
    if row["run_type"] == "model_save":
        return {
            **base,
            "model_path": row["model_path"],
            "duration_sec": row.get("duration"),
            "attack": row.get("attack"),
            "epsilon": row.get("epsilon"),
        }

    raise ValueError(f"Unknown run_type: {row['run_type']}")