import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN
from attacks.fgsm import fgsm_attack

model_path = 'models/cnn_mnist.pth'
batch_size = 64
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def get_data_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_fgsm(model, device, test_loader, epsilon):
    model.eval()
    correct = 0
    adv_examples = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        perturbed_data = fgsm_attack(model, device, images, labels, epsilon)
        output = model(perturbed_data)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()

        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((predicted[0].item(), labels[0].item(), adv_ex))

    final_acc = correct / len(test_loader.dataset)
    return final_acc, adv_examples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loader = get_data_loader(batch_size)

    print("Evaluating FGSM Attack...")
    print(f'|{"Epsilon":<12}|{"Test Accuracy":<12}|')
    print("|-|-|")

    for epsilon in epsilons:
        accuracy, _ = evaluate_fgsm(model, device, test_loader, epsilon)
        print(f'|{epsilon:<12.2f}|{accuracy*100:<.2f}%|')

if __name__ == "__main__":
    main()