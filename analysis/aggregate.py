# analysis/aggregate.py

import pandas as pd


def attack_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, count of accuracy per attack + epsilon across seeds."""
    return (
        df.groupby(["attack", "steps", "epsilon"])["accuracy"]
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
        df.groupby(["attack", "steps", "epsilon"])["accuracy"]
          .max()
          .reset_index()
          .rename(columns={"accuracy": "best_accuracy"})
    )


def training_final(df: pd.DataFrame) -> pd.DataFrame:
    """Final epoch stats per seed."""
    return (
        df.sort_values("epoch")
          .groupby("seed")
          .last()
          .reset_index()
        [["seed", "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]]
    )