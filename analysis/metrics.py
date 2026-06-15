# analysis/metrics.py

def add_attack_metrics(df):
    df = df.copy()

    if "attack" in df.columns:
        df["is_pgd"] = df["attack"] == "pgd"
        df["is_fgsm"] = df["attack"] == "fgsm"

    if "attack_params" in df.columns:
        df["steps"] = df["attack_params"].apply(
            lambda x: x.get("steps") if isinstance(x, dict) else None
        )

    return df


def add_runtime_metrics(df):
    df = df.copy()

    if "duration_sec" in df.columns:
        df["log_duration"] = df["duration_sec"]

    return df