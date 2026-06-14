import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
from attacks.pgd import pgd_attack

base_model_path = 'models/cnn_mnist.pth'
defense_model_path = 'models/cnn_mnist_fgsm_adv.pth'
batch_size = 64
epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
alpha = 0.01
iters = 40

def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_pgd(model, device, test_loader, epsilon):
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, device, images, labels, epsilon, alpha, iters)

        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = CNN().to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    base_model.eval()

    defense_model = CNN().to(device)
    defense_model.load_state_dict(torch.load(defense_model_path, map_location=device))
    defense_model.eval()

    test_loader = get_data_loader()

    print(f"PGD Attack — alpha={alpha}, steps={iters}\n")
    print(f"|{'Epsilon':<12}|{'Baseline':<14}|{'Defended':<14}|{'Delta':<10}|")
    print("|-"*4 + "|")

    for epsilon in epsilons:
        base_acc = evaluate_pgd(base_model, device, test_loader, epsilon)
        defense_acc = evaluate_pgd(defense_model, device, test_loader, epsilon)
        delta = defense_acc - base_acc

        sign = "+" if delta >= 0 else "-"
        print(f"| {epsilon:<10.2f} | {base_acc*100:<12.2f}% | {defense_acc*100:<12.2f}% | {sign}{delta*100:<8.2f}|")

if __name__ == "__main__":
    main()