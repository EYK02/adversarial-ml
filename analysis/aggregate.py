# analysis/aggregate.py

def summary_table(df):
    return df.groupby(
        ["attack", "epsilon"]
    )["value"].agg(["mean", "std", "count"]).reset_index()


def step_complexity(df):
    d = df[df["attack"] == "pgd"]

    return d.groupby("steps")["duration_sec"].agg(["mean", "std"])