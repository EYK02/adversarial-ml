# analysis/run_report.py

from pathlib import Path

from analysis.load_logs import load_all
from analysis.aggregate import (
    attack_summary,
    seed_variance,
    step_complexity,
    best_accuracy,
    training_final,
)
from analysis.plots import (
    plot_training_curves,
    plot_robustness_fgsm,
    plot_robustness_pgd,
    plot_seed_variance,
    plot_steps_scaling,
)

RESULTS_DIR = Path("results")
IMAGES_DIR  = RESULTS_DIR / "images"
CSV_DIR     = RESULTS_DIR / "csv"


def _setup_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, name):
    fig.savefig(IMAGES_DIR / name, dpi=150, bbox_inches="tight")
    fig.clf()


def main():
    _setup_dirs()
    dfs = load_all()

    train_df  = dfs["train"]
    attack_df = dfs["attack"]

    # --- training ---
    print("Training...")
    _save_fig(plot_training_curves(train_df), "training_curves.png")
    final = training_final(train_df)
    final.to_csv(CSV_DIR / "training_summary.csv", index=False)
    print(final.to_string(index=False))

    # --- robustness ---
    print("\nRobustness...")
    _save_fig(plot_robustness_fgsm(attack_df), "robustness_fgsm.png")
    _save_fig(plot_robustness_pgd(attack_df),  "robustness_pgd.png")

    # --- seed variance ---
    print("Seed variance...")
    _save_fig(plot_seed_variance(attack_df), "seed_variance.png")
    variance = seed_variance(attack_df)
    variance.to_csv(CSV_DIR / "seed_variance.csv", index=False)
    print(variance.to_string(index=False))

    # --- step scaling ---
    print("\nPGD step scaling...")
    _save_fig(plot_steps_scaling(attack_df), "pgd_step_scaling.png")
    step_complexity(attack_df).to_csv(CSV_DIR / "step_complexity.csv", index=False)

    # --- summary tables ---
    print("\nSummary tables...")
    attack_summary(attack_df).to_csv(CSV_DIR / "attack_summary.csv", index=False)
    best_accuracy(attack_df).to_csv(CSV_DIR / "best_accuracy.csv", index=False)

    print(f"\nDone. Outputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()