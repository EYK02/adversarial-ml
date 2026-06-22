# src/analysis/report.py

import argparse

from tools.analysis.load_logs import load_all
from tools.analysis.aggregate import (
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
from tools.analysis.plots import (
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
    if train_df.empty:
        print("  WARNING: no training logs found, skipping training curves")
    else:
        _save_fig(plot_training_curves(train_df), figures_dir / "training_curves.png")
        final = training_final(train_df)
        final.to_csv(metrics_dir / "training_summary.csv", index=False)
        print(final.to_string(index=False))

    # ── baseline robustness ──────────────────────────────────────
    print("\nBaseline robustness...")
    if attack_df.empty:
        print(" WARNING: no attack evaluation logs found, skipping baseline robustness")
    else:
        _save_fig(plot_robustness(attack_df), figures_dir / "robustness.png")
        attack_summary(attack_df).to_csv(metrics_dir / "attack_summary.csv",   index=False)
        best_accuracy(attack_df).to_csv(metrics_dir  / "best_accuracy.csv",    index=False)

    # ── seed variance ─────────────────────────────────────────────
    print("\nSeed variance...")
    if attack_df.empty:
        print("  WARNING: no attack evaluation logs found, skipping seed variance")
    else:
        seed_variance(attack_df).to_csv(metrics_dir / "seed_variance.csv", index=False)

        _save_fig(plot_fgsm_seed_variance(attack_df), figures_dir / "fgsm_seed_variance.png")

        if (attack_df["attack"] == "pgd").any():
            _save_fig(plot_pgd_seed_variance(attack_df), figures_dir / "pgd_seed_variance.png")
            _save_fig(plot_steps_scaling(attack_df),     figures_dir / "pgd_step_scaling.png")
            step_complexity(attack_df).to_csv(metrics_dir / "step_complexity.csv", index=False)
        else:
            print("  Skipping PGD-specific plots — no PGD eval data")

    # ── defense robustness ───────────────────────────────────────
    print("\nDefense robustness...")
    if defense_df.empty:
        print(" WARNING: no defense evaluation logs found, skipping defense robustness")
    else:
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
    if defense_df.empty:
        print(" WARNING: no defense evaluation logs found, skipping baseline vs defense comparison")
    else:
        _save_fig(plot_defense_vs_baseline(defense_df), figures_dir / "defense_vs_baseline.png")

    # ── cross-evaluation heatmaps ────────────────────────────────
    print("Cross-evaluation heatmaps...")
    if defense_df.empty:
        print(" WARNING: no defense evaluation logs found, skipping cross-evaluation heatmaps")
    else:
        for eps in cfg.epsilon_heatmap:
            _save_fig(
                plot_crosseval_heatmap(defense_df, epsilon=eps),
                figures_dir / f"crosseval_heatmap_eps{int(eps * 100):02d}.png",
            )

    # ── defense summary tables ───────────────────────────────────
    print("Defense summary tables...")
    if defense_df.empty:
        print(" WARNING: no defense evaluation logs found, skipping defense summary tables")
    else:
        defense_summary(defense_df).to_csv(
            metrics_dir / "defense_summary.csv",       index=False)
        defense_delta_summary(defense_df).to_csv(
            metrics_dir / "defense_delta_summary.csv", index=False)
        defense_seed_variance(defense_df).to_csv(
            metrics_dir / "defense_seed_variance.csv", index=False)

    # ── cross-evaluation pivot tables ────────────────────────────
    print("Cross-evaluation pivot tables...")
    if defense_df.empty:
        print(" WARNING: no defense evaluation logs found, skipping cross-evaluation pivot tables")
    else:
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
    parser.add_argument("--smoke-test",    action="store_true")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    cfg = load_experiment(
        args.experiment, 
        dry_run=args.dry_run, 
        smoke_test=args.smoke_test,
        run_name=args.run_name,
    )
    run_report(cfg)


if __name__ == "__main__":
    main()