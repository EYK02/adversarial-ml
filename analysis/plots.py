# analysis/plots.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ─────────────────────────────────────────
# BASELINE ROBUSTNESS
# ─────────────────────────────────────────

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

    fig, ax = plt.subplots(figsize=(8, 4))

    for steps in steps_list:
        d = pgd[pgd["steps"] == steps]
        grouped = d.groupby("epsilon")["accuracy"].agg(["mean", "std"]).reset_index()

        line, = ax.plot(grouped["epsilon"], grouped["mean"], marker="o", label=f"PGD-{int(steps)}")
        ax.fill_between(
            grouped["epsilon"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_title("PGD — accuracy vs epsilon (mean ± 1σ across seeds)")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────
# SEED VARIANCE
# ─────────────────────────────────────────

def _seed_variance_panel(ax, df, attack, steps=None):
    mask = df["attack"] == attack
    if steps is not None:
        mask &= df["steps"] == steps

    d = df[mask]
    means = d.groupby("epsilon")["accuracy"].mean()

    for seed, g in d.groupby("seed"):
        g = g.sort_values("epsilon")
        ax.scatter(g["epsilon"], g["accuracy"], s=15, alpha=0.6, label=f"seed {int(seed)}")

    ax.plot(means.index, means.values, color="black", linewidth=1.5, label="mean")

    label = attack.upper() if steps is None else f"PGD-{int(steps)}"
    ax.set_title(f"{label} — per-seed variance")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("accuracy (%)")


def plot_seed_variance_fgsm(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    _seed_variance_panel(ax, df, attack="fgsm")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def plot_seed_variance_pgd(df: pd.DataFrame) -> plt.Figure:
    pgd = df[df["attack"] == "pgd"]
    steps_list = sorted(pgd["steps"].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes_flat = axes.flatten()

    for ax, steps in zip(axes_flat, steps_list):
        _seed_variance_panel(ax, df, attack="pgd", steps=steps)

    # hide unused panels if fewer than 4
    for ax in axes_flat[len(steps_list):]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("PGD — per-seed variance")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────
# PGD STEP SCALING
# ─────────────────────────────────────────

def plot_steps_scaling(df: pd.DataFrame) -> plt.Figure:
    pgd = df[df["attack"] == "pgd"].copy()

    fig, ax = plt.subplots(figsize=(7, 4))

    epsilons = sorted(pgd["epsilon"].unique())
    if len(epsilons) > 4:
        idx = [0, len(epsilons) // 3, 2 * len(epsilons) // 3, -1]
        epsilons = [epsilons[i] for i in idx]

    for eps in epsilons:
        d = pgd[pgd["epsilon"] == eps]
        grouped = d.groupby("steps")["accuracy"].mean().reset_index()
        ax.plot(grouped["steps"], grouped["accuracy"], marker="o", label=f"ε={eps:.2f}")

    ax.set_title("PGD — accuracy vs steps (mean across seeds)")
    ax.set_xlabel("steps")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────
# DEFENSE ROBUSTNESS
# ─────────────────────────────────────────

def plot_defense_robustness(df: pd.DataFrame, eval_attack: str, eval_steps: int | None = None) -> plt.Figure:
    """
    For a given eval attack, plot defense_accuracy vs epsilon
    for each defense, with baseline overlaid.
    """
    mask = df["eval_attack"] == eval_attack
    if eval_steps is not None:
        mask &= df["eval_steps"] == eval_steps
    d = df[mask]

    eval_tag = eval_attack.upper() if eval_steps is None else f"PGD-{eval_steps}"

    fig, ax = plt.subplots(figsize=(8, 4))

    # baseline — same for all defenses, just take one
    baseline = (
        d.groupby("epsilon")["baseline_accuracy"]
         .mean()
         .reset_index()
    )
    ax.plot(baseline["epsilon"], baseline["baseline_accuracy"],
            marker="o", linestyle="--", color="black", label="baseline")

    # one line per defense config
    defense_configs = (
        d.groupby(["defense_attack", "defense_steps"])
         .size()
         .reset_index()[["defense_attack", "defense_steps"]]
    )

    for _, row in defense_configs.iterrows():
        d_attack, d_steps = row["defense_attack"], row["defense_steps"]

        mask2 = d["defense_attack"] == d_attack
        if pd.notna(d_steps):
            mask2 &= d["defense_steps"] == d_steps

        grouped = (
            d[mask2]
            .groupby("epsilon")["defense_accuracy"]
            .agg(["mean", "std"])
            .reset_index()
        )

        label = d_attack.upper() if pd.isna(d_steps) else f"PGD-{int(d_steps)}"
        line, = ax.plot(grouped["epsilon"], grouped["mean"], marker="o", label=label)
        ax.fill_between(
            grouped["epsilon"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_title(f"Defense robustness under {eval_tag} (mean ± 1σ across seeds)")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────
# CROSS-EVALUATION HEATMAP
# ─────────────────────────────────────────

def plot_crosseval_heatmap(df: pd.DataFrame, epsilon: float) -> plt.Figure:
    """
    At a fixed epsilon, grid of defense (rows) vs eval_attack (cols)
    showing mean defense_accuracy across seeds.
    """
    d = df[df["epsilon"].round(4) == round(epsilon, 4)]

    def defense_label(attack, steps):
        return attack.upper() if pd.isna(steps) else f"PGD-{int(steps)}"

    def eval_label(attack, steps):
        return attack.upper() if pd.isna(steps) else f"PGD-{int(steps)}"

    d = d.copy()
    d["defense_label"] = d.apply(lambda r: defense_label(r["defense_attack"], r["defense_steps"]), axis=1)
    d["eval_label"]    = d.apply(lambda r: eval_label(r["eval_attack"], r["eval_steps"]), axis=1)

    pivot = (
        d.groupby(["defense_label", "eval_label"])["defense_accuracy"]
         .mean()
         .unstack()
    )

    # row order: FGSM, PGD-5, PGD-10, PGD-20, PGD-40
    row_order = ["FGSM"] + [f"PGD-{s}" for s in [5, 10, 20, 40]]
    col_order = ["FGSM", "PGD-40"]
    pivot = pivot.reindex(index=[r for r in row_order if r in pivot.index],
                          columns=[c for c in col_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="accuracy (%)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("eval attack")
    ax.set_ylabel("defense")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color="black")

    ax.set_title(f"Cross-evaluation heatmap — ε={epsilon:.2f} (mean across seeds)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────
# BASELINE VS DEFENSE COMPARISON
# ─────────────────────────────────────────

def plot_defense_vs_baseline(df: pd.DataFrame, eval_attack: str, eval_steps: int | None = None) -> plt.Figure:
    """
    For each defense, plot baseline_accuracy and defense_accuracy
    vs epsilon side by side to show the lift.
    """
    mask = df["eval_attack"] == eval_attack
    if eval_steps is not None:
        mask &= df["eval_steps"] == eval_steps
    d = df[mask]

    eval_tag = eval_attack.upper() if eval_steps is None else f"PGD-{eval_steps}"

    defense_configs = (
        d.groupby(["defense_attack", "defense_steps"])
         .size()
         .reset_index()[["defense_attack", "defense_steps"]]
    )

    n = len(defense_configs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, defense_configs.iterrows()):
        d_attack, d_steps = row["defense_attack"], row["defense_steps"]

        mask2 = d["defense_attack"] == d_attack
        if pd.notna(d_steps):
            mask2 &= d["defense_steps"] == d_steps

        sub = d[mask2].groupby("epsilon")[["baseline_accuracy", "defense_accuracy"]].mean().reset_index()

        ax.plot(sub["epsilon"], sub["baseline_accuracy"],
                marker="o", linestyle="--", color="gray", label="baseline")
        ax.plot(sub["epsilon"], sub["defense_accuracy"],
                marker="o", label="defended")
        ax.fill_between(sub["epsilon"], sub["baseline_accuracy"], sub["defense_accuracy"],
                        alpha=0.1, color="green")

        label = d_attack.upper() if pd.isna(d_steps) else f"PGD-{int(d_steps)}"
        ax.set_title(label)
        ax.set_xlabel("epsilon")
        if ax is axes[0]:
            ax.set_ylabel("accuracy (%)")
        ax.legend(fontsize=8)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle(f"Baseline vs defense under {eval_tag} (mean across seeds)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────

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