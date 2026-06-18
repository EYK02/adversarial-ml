# analysis/aggregate.py

import pandas as pd


# Baseline Attack


def attack_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, count of accuracy per attack + steps + epsilon across seeds."""
    return (
        df.groupby(["attack", "steps", "epsilon"], dropna=False)["accuracy"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )


def seed_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, min, max of accuracy per attack + steps + epsilon across seeds."""
    return (
        df.groupby(["attack", "steps", "epsilon"], dropna=False)["accuracy"]
          .agg(["mean", "std", "min", "max"])
          .reset_index()
    )


def step_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of runtime per PGD step count."""
    pgd = df[df["attack"] == "pgd"]
    return (
        pgd.groupby("steps")["duration_sec"]
           .agg(["mean", "std"])
           .reset_index()
    )


def best_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Best accuracy per attack + steps + epsilon across seeds."""
    return (
        df.groupby(["attack", "steps", "epsilon"], dropna=False)["accuracy"]
          .max()
          .reset_index()
          .rename(columns={"accuracy": "best_accuracy"})
    )


# Training


def training_final(df: pd.DataFrame) -> pd.DataFrame:
    """Final epoch stats per seed."""
    return (
        df.sort_values("epoch")
          .groupby("seed")
          .last()
          .reset_index()
        [["seed", "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]]
    )


# Defense


def defense_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean, std, count of defense_accuracy per defense config
    + eval attack + epsilon across seeds.
    """
    return (
        df.groupby(
            ["defense_attack", "defense_steps", "defense_epsilon",
             "eval_attack", "eval_steps", "epsilon"],
            dropna=False
        )["defense_accuracy"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )


def defense_delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean delta (defense_accuracy - baseline_accuracy) per defense config
    + eval attack + epsilon. Positive = defense helped.
    """
    return (
        df.groupby(
            ["defense_attack", "defense_steps", "defense_epsilon",
             "eval_attack", "eval_steps", "epsilon"],
            dropna=False
        )["delta"]
          .agg(["mean", "std"])
          .reset_index()
          .rename(columns={"mean": "mean_delta", "std": "std_delta"})
    )


def defense_seed_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean, std, min, max of defense_accuracy per defense config
    + eval attack + epsilon across seeds.
    """
    return (
        df.groupby(
            ["defense_attack", "defense_steps",
             "eval_attack", "eval_steps", "epsilon"],
            dropna=False
        )["defense_accuracy"]
          .agg(["mean", "std", "min", "max"])
          .reset_index()
    )


def crosseval_pivot(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """
    At a fixed epsilon, pivot table of mean defense_accuracy:
    rows = defense config, cols = eval attack.
    Useful for quick cross-evaluation inspection in the console.
    """
    d = df[df["epsilon"].round(4) == round(epsilon, 4)].copy()

    def defense_label(attack, steps):
        return attack.upper() if pd.isna(steps) else f"PGD-{int(steps)}"

    def eval_label(attack, steps):
        return attack.upper() if pd.isna(steps) else f"PGD-{int(steps)}"

    d["defense_label"] = d.apply(
        lambda r: defense_label(r["defense_attack"], r["defense_steps"]), axis=1
    )
    d["eval_label"] = d.apply(
        lambda r: eval_label(r["eval_attack"], r["eval_steps"]), axis=1
    )

    baseline = (
        d.groupby("eval_label")["baseline_accuracy"]
         .mean()
    )
    baseline_row = pd.DataFrame([baseline], index=["Undefended"])
    
    pivot = (
        d.groupby(["defense_label", "eval_label"])["defense_accuracy"]
         .mean()
         .unstack()
    )
    pivot = pd.concat([baseline_row, pivot])

    row_order = ["Undefended", "FGSM"] + [f"PGD-{s}" for s in [5, 10, 20, 40]]
    col_order = ["FGSM", "PGD-40"]
    pivot = pivot.reindex(
        index=[r for r in row_order if r in pivot.index],
        columns=[c for c in col_order if c in pivot.columns]
    )
    return pivot