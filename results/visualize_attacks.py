import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import CNN
from attacks.registry import ATTACKS
from models.registry import MODELS
from utils import get_mnist_test_loader

epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

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
            color = 'green' if correct else 'red'
            ax.set_title(
                f'T:{true_label} P:{pred_label}',
                fontsize=8,
                color=color
            )

            if col == 0:
                ax.set_ylabel(f'ε={eps}', fontsize=9, rotation=0, labelpad=35)

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {save_path}')

def main():
    parser = argparse.ArgumentParser(description='Visualise adversarial attacks on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm', choices=ATTACKS.keys(),
                        help='Attack to visualise')
    parser.add_argument('--model', type=str, default='base', choices=MODELS.keys(),
                        help='Model to attack')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN().to(device)
    model.load_state_dict(torch.load(MODELS[args.model], map_location=device))
    model.eval()

    attack_fn = ATTACKS[args.attack]
    save_path = f'results/{args.attack}_{args.model}_visualization.png'
    title = f'{args.attack.upper()} Attack on {args.model} model (green=correct, red=misclassified)'

    test_loader = get_mnist_test_loader(batch_size=1)
    examples = get_examples(model, device, test_loader, attack_fn, epsilons)
    plot_examples(examples, epsilons, title, save_path)

if __name__ == '__main__':
    main()