# analysis/run_report.py

import os
from analysis.load_logs import load_all
from analysis.metrics import add_attack_metrics, add_runtime_metrics
from analysis.plots import plot_accuracy_vs_epsilon, plot_steps_vs_runtime
from analysis.aggregate import summary_table, step_complexity


def main():
    dfs = load_all()

    attack = add_attack_metrics(dfs["attack"])
    attack = add_runtime_metrics(attack)

    print("\n=== SUMMARY TABLE ===")
    print(summary_table(attack))

    print("\n=== PGD STEP COMPLEXITY ===")
    print(step_complexity(attack))

    for attack_name in attack["attack"].unique():
        plot_accuracy_vs_epsilon(attack, attack_name)

    plot_steps_vs_runtime(attack)

    print("\n=== BEST ROBUSTNESS PER EPSILON ===")
    print(
        attack.groupby(["epsilon", "attack"])["accuracy"]
        .max()
        .unstack()
    )


if __name__ == "__main__":
    main()