# analysis/plots.py

import matplotlib.pyplot as plt


def plot_accuracy_vs_epsilon(df, attack="pgd"):
    d = df[df["attack"] == attack]

    grouped = d.groupby("epsilon")["value"].mean().reset_index()

    plt.figure()
    plt.plot(grouped["epsilon"], grouped["value"])
    plt.title(f"Accuracy vs epsilon ({attack})")
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    plt.show()


def plot_steps_vs_runtime(df):
    d = df[df["attack"] == "pgd"]

    grouped = d.groupby("steps")["duration_sec"].mean().reset_index()

    plt.figure()
    plt.plot(grouped["steps"], grouped["duration_sec"])
    plt.title("PGD steps vs runtime")
    plt.xlabel("steps")
    plt.ylabel("seconds")
    plt.show()