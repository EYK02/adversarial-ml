# src/analysis/load_logs.py

import json
from pathlib import Path

import pandas as pd

from src.utils.schema import normalize

PARAM_COLS = ["attack_params", "defense_params", "eval_params"]


def _parse_params(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for col in PARAM_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_params)

    return df


def load_all(logs_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Loads all experiment logs from logs_dir and returns
    normalized DataFrames keyed by log type.
    """
    sources = {
        "train":     logs_dir / "standard.jsonl",
        "attack":    logs_dir / "eval_attack.jsonl",
        "adv_train": logs_dir / "adv_training.jsonl",
        "defense":   logs_dir / "eval_robustness.jsonl",
    }

    dfs = {}
    for key, path in sources.items():
        raw            = load_jsonl(path)
        dfs[f"{key}_raw"] = raw
        dfs[key]       = normalize(raw) if not raw.empty else pd.DataFrame()

    return dfs