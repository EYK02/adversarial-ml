# src/visualization/examples.py

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from src.attacks.registry import get_attack_fn
from src.models.factory import load_model
from src.utils.config import load_experiment
from src.utils.seed import get_device, set_seed
from src.datasets.mnist import get_test_loader


VALID_ATTACKS    = ["fgsm", "pgd"]
DEFAULT_EPSILONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def get_examples(model, device, test_loader, attack_fn, epsilons, n=5):
    examples = {eps: [] for eps in epsilons}

    for images, labels in test_loader:
        if all(len(v) >= n for v in examples.values()):
            break

        images, labels = images.to(device), labels.to(device)

        for eps in epsilons:
            if len(examples[eps]) >= n:
                continue

            perturbed  = attack_fn(model, device, images.clone(), labels, eps)
            _, pred    = torch.max(model(perturbed), 1)
            img        = perturbed.squeeze().detach().cpu().numpy()
            examples[eps].append((labels.item(), pred.item(), img))

    return examples


def plot_examples(examples, epsilons, title, save_path, n=5):
    fig, axes = plt.subplots(
        len(epsilons), n,
        figsize=(n * 2, len(epsilons) * 2),
        squeeze=False,
    )

    for row, eps in enumerate(epsilons):
        for col, (true_label, pred_label, img) in enumerate(examples[eps][:n]):
            ax = axes[row][col]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(
                f'T:{true_label} P:{pred_label}',
                fontsize=8,
                color='green' if true_label == pred_label else 'red',
            )
            if col == 0:
                ax.set_ylabel(f'ε={eps:.2f}', fontsize=9, rotation=0, labelpad=35)

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualise adversarial attacks on MNIST')
    parser.add_argument('--config',     type=str, required=True,
                        help='Experiment config yaml (for dataset/model config)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint .pth')
    parser.add_argument('--attack',     type=str, default='fgsm', choices=VALID_ATTACKS)
    parser.add_argument('--steps',      type=int, default=10,
                        help='PGD step count (PGD only)')
    parser.add_argument('--restarts',   type=int, default=1,
                        help='PGD restarts (PGD only)')
    parser.add_argument('--epsilons',   type=float, nargs='+', default=DEFAULT_EPSILONS)
    parser.add_argument('--n',          type=int, default=5,
                        help='Example images per epsilon row')
    parser.add_argument('--seed',       type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    cfg   = load_experiment(args.config)
    model = load_model(args.model_path, device, cfg.model)

    if args.attack == 'fgsm':
        attack_fn, _ = get_attack_fn('fgsm')
        attack_tag   = 'fgsm'
    else:
        # alpha resolved per-epsilon inside get_examples via the attack_fn closure,
        # so we use a representative epsilon for the tag but resolve alpha at call time
        attack_tag = f"pgd{args.steps}"

        def attack_fn(model, device, images, labels, eps):
            alpha = 2.5 * eps / args.steps if eps > 0 else 0.0
            fn, _ = get_attack_fn('pgd', steps=args.steps, alpha=alpha, restarts=args.restarts)
            return fn(model, device, images, labels, eps)

    model_tag = os.path.splitext(os.path.basename(args.model_path))[0]
    save_path = f"artifacts/images/{attack_tag}_{model_tag}_visualization.png"
    title     = f"{attack_tag.upper()} on {model_tag} (green=correct, red=misclassified)"

    test_loader = get_test_loader(cfg.dataset, batch_size=1)
    examples    = get_examples(model, device, test_loader, attack_fn, args.epsilons, n=args.n)
    plot_examples(examples, args.epsilons, title, save_path, n=args.n)


if __name__ == '__main__':
    main()