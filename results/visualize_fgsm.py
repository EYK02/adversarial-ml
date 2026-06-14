import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
from attacks.fgsm import fgsm_attack

model_path = 'models/cnn_mnist.pth'
epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
save_path = 'results/fgsm_visualization.png'

# TODO: Refactor to generalize for FGSM, PGD, and other attacks with minimal code duplication.


def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    return DataLoader(test_dataset, batch_size=1, shuffle=True)

def get_examples(model, device, test_loader, epsilons, n=5):
    examples = {eps: [] for eps in epsilons}

    for images, labels in test_loader:
        if all(len(v) >= n for v in examples.values()):
            break

        images, labels = images.to(device), labels.to(device)

        for eps in epsilons:
            if len(examples[eps]) >= n:
                continue

            perturbed = fgsm_attack(model, device, images.clone(), labels, eps)
            output = model(perturbed)
            _, predicted = torch.max(output, 1)

            img = perturbed.squeeze().detach().cpu().numpy()
            examples[eps].append((labels.item(), predicted.item(), img))

    return examples

def plot_examples(examples, epsilons, n=5):
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

    plt.suptitle('FGSM Attack — Original vs Perturbed (green=correct, red=misclassified)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {save_path}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = get_test_loader()
    examples = get_examples(model, device, test_loader, epsilons)
    plot_examples(examples, epsilons)

if __name__ == '__main__':
    main()