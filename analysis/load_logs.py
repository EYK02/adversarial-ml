# analysis/load_logs.py

import json
import pandas as pd
from analysis.normalize import normalize


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_all():
    """
    Loads all experiment logs and returns:
    - raw views (for debugging)
    - normalized views (for analysis)
    """

    dfs = {}

    # ---------------- raw logs ----------------
    dfs["train_raw"] = load_jsonl("results/jsonl/training.jsonl")
    dfs["attack_raw"] = load_jsonl("results/jsonl/attack_eval.jsonl")
    dfs["adv_train_raw"] = load_jsonl("results/jsonl/adv_training.jsonl")
    dfs["defense_raw"] = load_jsonl("results/jsonl/defense_eval.jsonl")

    # ---------------- normalized logs ----------------
    dfs["train"] = normalize(dfs["train_raw"])
    dfs["attack"] = normalize(dfs["attack_raw"])
    dfs["adv_train"] = normalize(dfs["adv_train_raw"])
    dfs["defense"] = normalize(dfs["defense_raw"])

    return dfs