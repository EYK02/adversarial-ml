# analysis/report.py

from pathlib import Path

from analysis.load_logs import load_all
from analysis.aggregate import (
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
from analysis.plots import (
    plot_training_curves,
    plot_robustness,
    plot_fgsm_seed_variance,
    plot_pgd_seed_variance,
    plot_steps_scaling,
    plot_defense_robustness,
    plot_crosseval_heatmap,
    plot_defense_vs_baseline,
)

ARTIFACTS_DIR = Path("artifacts")  
IMAGES_DIR    = ARTIFACTS_DIR / "images"
CSV_DIR       = ARTIFACTS_DIR / "csv"


def _setup_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, name):
    fig.savefig(IMAGES_DIR / name, dpi=150, bbox_inches="tight")
    fig.clf()


def main():
    _setup_dirs()
    dfs = load_all()
    

    train_df   = dfs["train"]
    attack_df  = dfs["attack"]
    defense_df = dfs["defense"]

    # ── training ────────────────────────────────────────────────
    print("Training...")
    _save_fig(plot_training_curves(train_df), "training_curves.png")
    final = training_final(train_df)
    final.to_csv(CSV_DIR / "training_summary.csv", index=False)
    print(final.to_string(index=False))

    # ── baseline robustness ──────────────────────────────────────
    print("\nBaseline robustness...")
    _save_fig(plot_robustness(attack_df), "robustness.png")
    attack_summary(attack_df).to_csv(CSV_DIR / "attack_summary.csv", index=False)
    best_accuracy(attack_df).to_csv(CSV_DIR / "best_accuracy.csv", index=False)

    # ── seed variance ────────────────────────────────────────────
    print("\nFGSM seed variance...")
    fgsm_fig = plot_fgsm_seed_variance(attack_df)
    _save_fig(fgsm_fig, "fgsm_seed_variance.png")

    print("\nPGD seed variance...")
    pgd_fig = plot_pgd_seed_variance(attack_df)
    _save_fig(pgd_fig, "pgd_seed_variance.png")

    variance = seed_variance(attack_df)
    variance.to_csv(CSV_DIR / "seed_variance.csv", index=False)
    print(variance.to_string(index=False))

    # ── step scaling ─────────────────────────────────────────────
    print("\nPGD step scaling...")
    _save_fig(plot_steps_scaling(attack_df), "pgd_step_scaling.png")
    step_complexity(attack_df).to_csv(CSV_DIR / "step_complexity.csv", index=False)

    # ── defense (skip if no data yet) ───────────────────────────
    print("\nDefense robustness...")
    _save_fig(
        plot_defense_robustness(defense_df, eval_attack="fgsm"),
        "defense_robustness_vs_fgsm.png"
    )
    _save_fig(
        plot_defense_robustness(defense_df, eval_attack="pgd", eval_steps=40),
        "defense_robustness_vs_pgd40.png"
    )

    print("Baseline vs defense comparison...")
    _save_fig(
        plot_defense_vs_baseline(defense_df),
        "defense_vs_baseline.png"
    )

    print("Cross-evaluation heatmap...")
    for eps in [0.0, 0.1, 0.2, 0.3]:
        _save_fig(
            plot_crosseval_heatmap(defense_df, epsilon=eps),
            f"crosseval_heatmap_eps{int(eps*100):02d}.png"
        )

    print("Defense summary tables...")
    defense_summary(defense_df).to_csv(CSV_DIR / "defense_summary.csv", index=False)
    defense_delta_summary(defense_df).to_csv(CSV_DIR / "defense_delta_summary.csv", index=False)
    defense_seed_variance(defense_df).to_csv(CSV_DIR / "defense_seed_variance.csv", index=False)

    print("Cross-evaluation pivot tables...")
    for eps in [0.0, 0.1, 0.2, 0.3]:
        pivot = crosseval_pivot(defense_df, epsilon=eps)
        pivot.to_csv(CSV_DIR / f"crosseval_pivot_eps{int(eps*100):02d}.csv", index_label="defense_label")
        print(f"\n  Cross-eval at ε={eps:.1f}:")
        print(pivot.to_string())

    print(f"\nDone. Outputs in {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()