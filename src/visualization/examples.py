# src/visualization/examples.py

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attacks.registry import build_attack
from src.models.factory import load_model
from src.utils.config import load_experiment, AttackConfig
from src.utils.seed import set_seed, get_device
from src.datasets.mnist import get_test_loader


VALID_ATTACKS = ["fgsm", "pgd"]
DEFAULT_EPSILONS = [0.0, 0.15, 0.30, 0.45]


# ─────────────────────────────────────────────────────────────
# attack construction (single shared function)
# ─────────────────────────────────────────────────────────────

def build_attack_fn(attack: str, steps: int, restarts: int):
    """
    Returns:
        attack_fn: callable(model, device, images, labels, eps)
        tag: str
    """

    if attack == "fgsm":
        spec = AttackConfig(
            name="fgsm",
            epsilon=None,
            steps=None,
            alpha=None,
            restarts=None,
        )
        fn, _ = build_attack(spec)
        return fn, "fgsm"

    if attack == "pgd":
        spec = AttackConfig(
            name="pgd",
            epsilon=None,   # epsilon is runtime-swept in visualization
            steps=steps,
            alpha=None,     # resolved inside registry
            restarts=restarts,
        )
        fn, _ = build_attack(spec)
        return fn, f"pgd{steps}"

    raise ValueError(f"Unknown attack: {attack}")


# ─────────────────────────────────────────────────────────────
# sample extraction
# ─────────────────────────────────────────────────────────────

def get_examples(model, device, test_loader, attack_fn, epsilons, n=5):
    examples = {eps: [] for eps in epsilons}

    for images, labels in test_loader:
        if all(len(v) >= n for v in examples.values()):
            break

        images, labels = images.to(device), labels.to(device)

        for eps in epsilons:
            if len(examples[eps]) >= n:
                continue

            perturbed = attack_fn(model, device, images.clone(), labels, eps)

            preds = model(perturbed).argmax(dim=1)

            img = perturbed.squeeze().detach().cpu().numpy()

            examples[eps].append(
                (labels.item(), preds.item(), img)
            )

    return examples


# ─────────────────────────────────────────────────────────────
# plotting
# ─────────────────────────────────────────────────────────────

def plot_examples(examples, epsilons, title, save_path, n=5):
    fig, axes = plt.subplots(
        len(epsilons),
        n,
        figsize=(n * 2, len(epsilons) * 2),
        squeeze=False,
    )

    for row, eps in enumerate(epsilons):
        for col, (true_label, pred_label, img) in enumerate(examples[eps][:n]):
            ax = axes[row][col]

            ax.imshow(img, cmap="gray")
            ax.axis("off")

            ax.set_title(
                f"T:{true_label} P:{pred_label}",
                fontsize=8,
                color="green" if true_label == pred_label else "red",
            )

            if col == 0:
                ax.set_ylabel(
                    f"ε={eps:.2f}",
                    fontsize=9,
                    rotation=0,
                    labelpad=35,
                )

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise adversarial attacks on MNIST"
    )

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--attack",
        type=str,
        default="fgsm",
        choices=VALID_ATTACKS,
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=DEFAULT_EPSILONS,
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # ── setup ────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_device()

    cfg = load_experiment(args.config)
    model = load_model(args.model_path, device, cfg.model)
    model.eval()

    # ── attack ───────────────────────────────────────────────
    attack_fn, attack_tag = build_attack_fn(
        args.attack,
        args.steps,
        args.restarts,
    )

    # ── output path ──────────────────────────────────────────
    model_tag = Path(args.model_path).stem

    save_path = cfg.paths.figures / f"{attack_tag}_{model_tag}_visualization.png"

    title = (
        f"{attack_tag.upper()} on {model_tag} "
        f"(green=correct, red=misclassified)"
    )

    # ── data ────────────────────────────────────────────────
    test_loader = get_test_loader(cfg.dataset, batch_size=1)

    examples = get_examples(
        model,
        device,
        test_loader,
        attack_fn,
        args.epsilons,
        n=args.n,
    )

    # ── plot ────────────────────────────────────────────────
    plot_examples(
        examples,
        args.epsilons,
        title,
        str(save_path),
        n=args.n,
    )


if __name__ == "__main__":
    main()