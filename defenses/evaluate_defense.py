import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

ATTACKS = {
    'fgsm': lambda model, device, data, target, eps: fgsm_attack(
        model, device, data, target, eps
    ),
    'pgd5': lambda model, device, data, target, eps: pgd_attack(
        model, device, data, target, eps, alpha=0.01, iters=5
    ),
    'pgd10': lambda model, device, data, target, eps: pgd_attack(
        model, device, data, target, eps, alpha=0.01, iters=10
    ),
    'pgd20': lambda model, device, data, target, eps: pgd_attack(
        model, device, data, target, eps, alpha=0.01, iters=20
    ),
}

DEFENDED_MODELS = {
    'fgsm': 'models/cnn_mnist_fgsm_adv.pth',
    'pgd5': 'models/cnn_mnist_pgd5_adv.pth',
    'pgd10': 'models/cnn_mnist_pgd10_adv.pth',
    'pgd20': 'models/cnn_mnist_pgd20_adv.pth',
}

base_model_path = 'models/cnn_mnist.pth'
batch_size = 64
epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def evaluate_model(model, device, test_loader, attack_fn, epsilon):
    model.eval()
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = attack_fn(model, device, data, target, epsilon)
        output = model(adv_data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial defense on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm', choices=ATTACKS.keys(),
                        help='Attack to evaluate against')
    parser.add_argument('--defense', type=str, default='fgsm', choices=DEFENDED_MODELS.keys(),
                        help='Defended model to evaluate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = CNN().to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    base_model.eval()

    defense_model = CNN().to(device)
    defense_model.load_state_dict(torch.load(DEFENDED_MODELS[args.defense], map_location=device))
    defense_model.eval()

    attack_fn = ATTACKS[args.attack]
    test_loader = get_data_loader()

    print(f"Attack: {args.attack} | Defense: {args.defense}\n")
    print(f"|{'Epsilon':<12}|{'Baseline':<14}|{'Defended':<14}|{'Delta':<10}|")
    print("|-" * 4 + "|")

    for epsilon in epsilons:
        base_acc = evaluate_model(base_model, device, test_loader, attack_fn, epsilon)
        def_acc = evaluate_model(defense_model, device, test_loader, attack_fn, epsilon)
        delta = def_acc - base_acc
        sign = "+" if delta >= 0 else ""
        base_str = f"{base_acc:.2f}%"
        def_str = f"{def_acc:.2f}%"
        delta_str = f"{sign}{abs(delta):.2f}%"
        print(f"| {epsilon:<10.2f} | {base_str:<12} | {def_str:<12} | {delta_str:<10}|")

if __name__ == '__main__':
    main()