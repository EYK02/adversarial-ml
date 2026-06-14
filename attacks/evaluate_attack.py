import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
from attacks.registry import ATTACKS

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

def evaluate(model, device, test_loader, attack_fn, epsilon):
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
    parser = argparse.ArgumentParser(description='Evaluate adversarial attack on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm', choices=ATTACKS.keys(),
                        help='Attack to evaluate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN().to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    model.eval()

    attack_fn = ATTACKS[args.attack]
    test_loader = get_data_loader()

    print(f"Evaluating {args.attack.upper()} attack on base model\n")
    print(f"|{'Epsilon':<12}|{'Accuracy':<12}|")
    print("|-|-|")

    for epsilon in epsilons:
        acc = evaluate(model, device, test_loader, attack_fn, epsilon)
        print(f"|{epsilon:<12.2f}|{acc:<.2f}%|")

if __name__ == '__main__':
    main()