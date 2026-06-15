# analysis/load_logs.py

import json
import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_all():
    dfs = {}

    dfs["train"] = load_jsonl("results/jsonl/training.jsonl")
    dfs["attack"] = load_jsonl("results/jsonl/attack_eval.jsonl")
    dfs["adv_train"] = load_jsonl("results/jsonl/adv_training.jsonl")
    dfs["defense"] = load_jsonl("results/jsonl/defense_eval.jsonl")
    
    return dfs