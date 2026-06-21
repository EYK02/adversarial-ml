# src/analysis/report.py

import argparse

from src.analysis.load_logs import load_all
from src.analysis.aggregate import (
    attack_summary,
    seed_variance,
    step_complexity,
    best_accuracy,
    training_final,
    defense_summary,
    defense_delta_summary,
    defense_seed_variance,
    crosseval_pivot,
)
from src.analysis.plots import (
    plot_training_curves,
    plot_robustness,
    plot_fgsm_seed_variance,
    plot_pgd_seed_variance,
    plot_steps_scaling,
    plot_defense_robustness,
    plot_crosseval_heatmap,
    plot_defense_vs_baseline,
)
from src.utils.config import load_experiment, ExperimentConfig


def _save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.clf()


def run_report(cfg: ExperimentConfig) -> None:
    figures_dir = cfg.paths.figures
    metrics_dir = cfg.paths.metrics

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dfs        = load_all(cfg.paths.logs)
    train_df   = dfs["train"]
    attack_df  = dfs["attack"]
    defense_df = dfs["defense"]

    # ── training ─────────────────────────────────────────────────
    print("Training curves...")
    _save_fig(plot_training_curves(train_df), figures_dir / "training_curves.png")
    final = training_final(train_df)
    final.to_csv(metrics_dir / "training_summary.csv", index=False)
    print(final.to_string(index=False))

    # ── baseline robustness ──────────────────────────────────────
    print("\nBaseline robustness...")
    _save_fig(plot_robustness(attack_df), figures_dir / "robustness.png")
    attack_summary(attack_df).to_csv(metrics_dir / "attack_summary.csv",   index=False)
    best_accuracy(attack_df).to_csv(metrics_dir  / "best_accuracy.csv",    index=False)

    # ── seed variance ────────────────────────────────────────────
    print("\nFGSM seed variance...")
    _save_fig(plot_fgsm_seed_variance(attack_df), figures_dir / "fgsm_seed_variance.png")

    print("\nPGD seed variance...")
    _save_fig(plot_pgd_seed_variance(attack_df),  figures_dir / "pgd_seed_variance.png")

    seed_variance(attack_df).to_csv(metrics_dir / "seed_variance.csv", index=False)

    # ── step scaling ─────────────────────────────────────────────
    print("\nPGD step scaling...")
    _save_fig(plot_steps_scaling(attack_df), figures_dir / "pgd_step_scaling.png")
    step_complexity(attack_df).to_csv(metrics_dir / "step_complexity.csv", index=False)

    # ── defense robustness ───────────────────────────────────────
    print("\nDefense robustness...")
    for eval_attack, eval_steps, tag in [
        ("fgsm", None, "fgsm"),
        ("pgd",  10,   "pgd10"),
        ("pgd",  40,   "pgd40"),
    ]:
        _save_fig(
            plot_defense_robustness(defense_df, eval_attack=eval_attack, eval_steps=eval_steps),
            figures_dir / f"defense_robustness_vs_{tag}.png",
        )

    # ── baseline vs defense comparison ───────────────────────────
    print("Baseline vs defense comparison...")
    _save_fig(plot_defense_vs_baseline(defense_df), figures_dir / "defense_vs_baseline.png")

    # ── cross-evaluation heatmaps ────────────────────────────────
    print("Cross-evaluation heatmaps...")
    for eps in cfg.epsilon_heatmap:
        _save_fig(
            plot_crosseval_heatmap(defense_df, epsilon=eps),
            figures_dir / f"crosseval_heatmap_eps{int(eps * 100):02d}.png",
        )

    # ── defense summary tables ───────────────────────────────────
    print("Defense summary tables...")
    defense_summary(defense_df).to_csv(
        metrics_dir / "defense_summary.csv",       index=False)
    defense_delta_summary(defense_df).to_csv(
        metrics_dir / "defense_delta_summary.csv", index=False)
    defense_seed_variance(defense_df).to_csv(
        metrics_dir / "defense_seed_variance.csv", index=False)

    # ── cross-evaluation pivot tables ────────────────────────────
    print("Cross-evaluation pivot tables...")
    for eps in cfg.epsilon_heatmap:
        pivot = crosseval_pivot(defense_df, epsilon=eps)
        pivot.to_csv(
            metrics_dir / f"crosseval_pivot_eps{int(eps * 100):02d}.csv",
            index_label="defense_label",
        )
        print(f"\n  Cross-eval at ε={eps:.2f}:")
        print(pivot.to_string())

    print(f"\nDone. Figures → {figures_dir}  Metrics → {metrics_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    cfg = load_experiment(args.experiment, dry_run=args.dry_run)
    run_report(cfg)


if __name__ == "__main__":
    main()