# analysis/plots.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_training_curves(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for seed, group in df.groupby("seed"):
        g = group.sort_values("epoch")
        axes[0].plot(g["epoch"], g["train_loss"], label=f"seed {seed}")
        axes[1].plot(g["epoch"], g["test_accuracy"], label=f"seed {seed}")

    axes[0].set_title("Train loss per epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Test accuracy per epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_robustness_fgsm(df: pd.DataFrame) -> plt.Figure:
    fgsm = df[df["attack"] == "fgsm"]
    grouped = fgsm.groupby("epsilon")["accuracy"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped["epsilon"], grouped["mean"], marker="o")
    ax.fill_between(
        grouped["epsilon"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.2,
    )
    ax.set_title("FGSM — accuracy vs epsilon (mean ± 1σ across seeds)")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("accuracy (%)")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    return fig


def plot_robustness_pgd(df: pd.DataFrame) -> plt.Figure:
    pgd = df[df["attack"] == "pgd"]
    steps_list = sorted(pgd["steps"].dropna().unique())

    fig, axes = plt.subplots(1, len(steps_list), figsize=(4 * len(steps_list), 4), sharey=True)
    if len(steps_list) == 1:
        axes = [axes]

    for ax, steps in zip(axes, steps_list):
        d = pgd[pgd["steps"] == steps]
        grouped = d.groupby("epsilon")["accuracy"].agg(["mean", "std"]).reset_index()

        ax.plot(grouped["epsilon"], grouped["mean"], marker="o")
        ax.fill_between(
            grouped["epsilon"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
        )
        ax.set_title(f"PGD-{int(steps)}")
        ax.set_xlabel("epsilon")
        if ax is axes[0]:
            ax.set_ylabel("accuracy (%)")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle("PGD — accuracy vs epsilon (mean ± 1σ across seeds)")
    fig.tight_layout()
    return fig


def plot_seed_variance(df: pd.DataFrame) -> plt.Figure:
    configs = (
        df.groupby(["attack", "steps"], dropna=False)
          .size()
          .reset_index()[["attack", "steps"]]
    )

    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, configs.iterrows()):
        attack, steps = row["attack"], row["steps"]

        mask = df["attack"] == attack
        if pd.notna(steps):
            mask &= df["steps"] == steps

        d = df[mask]
        means = d.groupby("epsilon")["accuracy"].mean()

        for seed, g in d.groupby("seed"):
            g = g.sort_values("epsilon")
            ax.scatter(g["epsilon"], g["accuracy"], s=15, alpha=0.6, label=f"seed {int(seed)}")

        ax.plot(means.index, means.values, color="black", linewidth=1.5, label="mean")

        label = attack.upper() if pd.isna(steps) else f"PGD-{int(steps)}"
        ax.set_title(f"{label} — per-seed variance")
        ax.set_xlabel("epsilon")
        if ax is axes[0]:
            ax.set_ylabel("accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def plot_steps_scaling(df: pd.DataFrame) -> plt.Figure:
    pgd = df[df["attack"] == "pgd"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: accuracy vs steps at representative epsilons
    epsilons = sorted(pgd["epsilon"].unique())
    if len(epsilons) > 4:
        idx = [0, len(epsilons) // 3, 2 * len(epsilons) // 3, -1]
        epsilons = [epsilons[i] for i in idx]

    for eps in epsilons:
        d = pgd[pgd["epsilon"] == eps]
        grouped = d.groupby("steps")["accuracy"].mean().reset_index()
        axes[0].plot(grouped["steps"], grouped["accuracy"], marker="o", label=f"ε={eps:.2f}")

    axes[0].set_title("PGD — accuracy vs steps (mean across seeds)")
    axes[0].set_xlabel("steps")
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(fontsize=8)

    # right: runtime vs steps
    grouped = pgd.groupby("steps")["duration_sec"].agg(["mean", "std"]).reset_index()
    axes[1].bar(grouped["steps"].astype(str), grouped["mean"],
                yerr=grouped["std"], capsize=4, color="steelblue")
    axes[1].set_title("PGD — runtime vs steps (mean ± 1σ across seeds)")
    axes[1].set_xlabel("steps")
    axes[1].set_ylabel("seconds")

    fig.tight_layout()
    return fig