import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
from attacks.fgsm import fgsm_attack

base_model_path = 'models/cnn_mnist.pth'
defense_model_path = 'models/cnn_mnist_fgsm_adv.pth'
batch_size = 64
epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def get_data_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_model(model, device, test_loader, epsilon):
    model.eval()
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = fgsm_attack(model, device, data, target, epsilon)
        output = model(adv_data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    base_model = CNN().to(device)
    defense_model = CNN().to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    defense_model.load_state_dict(torch.load(defense_model_path, map_location=device))

    test_loader = get_data_loader(batch_size)

    print(f"{'Epsilon':<12} {'Baseline':<14} {'Defended':<14} {'Delta':<10}")
    print("-" * 50)

    for epsilon in epsilons:
        baseline_acc = evaluate_model(base_model, device, test_loader, epsilon)
        defended_acc = evaluate_model(defense_model, device, test_loader, epsilon)
        delta = defended_acc - baseline_acc
        sign = "+" if delta >= 0 else ""

        base_str = f"{baseline_acc:.2f}%"
        defended_str = f"{defended_acc:.2f}%"
        print(f"{epsilon:<12} {base_str:<14} {defended_str:<14} {sign}{delta:<10.2f}")

if __name__ == "__main__":
    main()