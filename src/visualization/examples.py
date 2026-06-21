# tools/visualize_attacks.py

import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attacks.registry import get_attack_fn
from models.factory import load_model
from src.datasets.mnist import get_mnist_test_loader
from utils.seed import set_seed
from old.src.utils.config import EPSILONS

VALID_ATTACKS = ["fgsm", "pgd"]


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
            output = model(perturbed)
            _, predicted = torch.max(output, 1)

            img = perturbed.squeeze().detach().cpu().numpy()
            examples[eps].append((labels.item(), predicted.item(), img))

    return examples


def plot_examples(examples, epsilons, title, save_path, n=5):
    fig, axes = plt.subplots(
        len(epsilons), n,
        figsize=(n * 2, len(epsilons) * 2)
    )

    for row, eps in enumerate(epsilons):
        for col, (true_label, pred_label, img) in enumerate(examples[eps][:n]):
            ax = axes[row][col]
            ax.imshow(img, cmap='gray')
            ax.axis('off')

            correct = true_label == pred_label
            ax.set_title(
                f'T:{true_label} P:{pred_label}',
                fontsize=8,
                color='green' if correct else 'red'
            )

            if col == 0:
                ax.set_ylabel(f'ε={eps}', fontsize=9, rotation=0, labelpad=35)

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualise adversarial attacks on MNIST')
    parser.add_argument('--attack',  type=str, default='fgsm', choices=VALID_ATTACKS,
                        help='Attack to visualise')
    parser.add_argument('--steps',   type=int, default=None,
                        help='PGD step count (PGD only)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model .pth file')
    parser.add_argument('--seed',    type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_path, device)
    model.eval()

    attack_fn, _ = get_attack_fn(args.attack, steps=args.steps)

    attack_tag = args.attack if args.steps is None else f"{args.attack}{args.steps}"
    model_tag  = args.model_path.replace("models/", "").replace(".pth", "")
    save_path  = f"artifacts/images/{attack_tag}_{model_tag}_visualization.png"
    title      = f"{attack_tag.upper()} on {model_tag} (green=correct, red=misclassified)"

    test_loader = get_mnist_test_loader(batch_size=1)
    examples    = get_examples(model, device, test_loader, attack_fn, EPSILONS)
    plot_examples(examples, EPSILONS, title, save_path)


if __name__ == '__main__':
    main()