# analysis/normalize.py

import pandas as pd


def _flatten_attack_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts:
        attack_params = {"steps": 20}
    into:
        attack_steps = 20
    """

    if "attack_params" not in df.columns:
        df["attack_steps"] = None
        return df

    def extract_steps(x):
        if isinstance(x, dict):
            return x.get("steps", None)
        return None

    df["attack_steps"] = df["attack_params"].apply(extract_steps)
    df = df.drop(columns=["attack_params"])

    return df


def _ensure_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict numeric typing for analysis consistency.
    """

    numeric_cols = ["epsilon", "duration_sec", "attack_steps", "seed"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "seed" in df.columns:
        df["seed"] = df["seed"].astype("Int64")

    if "attack_steps" in df.columns:
        df["attack_steps"] = df["attack_steps"].astype("Int64")

    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical normalization step:
    - flatten nested structures
    - enforce schema-compatible flat structure
    - ensure correct dtypes
    """

    df = df.copy()

    # 1. flatten nested attack params
    df = _flatten_attack_params(df)

    # 2. ensure numeric consistency
    df = _ensure_numeric_types(df)

    return df