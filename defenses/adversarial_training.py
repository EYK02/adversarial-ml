import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.registry import ATTACKS
from models.registry import MODELS

base_model_path = 'models/cnn_mnist.pth'
batch_size = 64
epochs = 5
learning_rate = 0.001

def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_adversarial(model, device, train_loader, optimizer, criterion, attack_fn, epsilon):
    model.train()

    total_loss = 0
    correct_clean = 0
    correct_adv = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        adv_data = attack_fn(model, device, data, target, epsilon)

        combined_data = torch.cat((data, adv_data), dim=0)
        combined_target = torch.cat((target, target), dim=0)

        optimizer.zero_grad()
        output = model(combined_data)
        loss = criterion(output, combined_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted_clean = torch.max(output[:data.size(0)].data, 1)
        _, predicted_adv = torch.max(output[data.size(0):].data, 1)

        correct_clean += (predicted_clean == target).sum().item()
        correct_adv += (predicted_adv == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    clean_acc = 100 * correct_clean / len(train_loader.dataset)
    adv_acc = 100 * correct_adv / len(train_loader.dataset)
    return avg_loss, clean_acc, adv_acc

def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / len(test_loader.dataset)
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Adversarial training on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm', choices=ATTACKS.keys(),
                        help='Attack to train against')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Perturbation budget for adversarial training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_data_loaders()

    defense_model = CNN().to(device)
    defense_model.load_state_dict(torch.load(base_model_path, map_location=device))

    optimizer = optim.Adam(defense_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    attack_fn = ATTACKS[args.attack]
    save_path = MODELS[args.attack]

    print(f"Adversarial training — attack={args.attack}, epsilon={args.epsilon}\n")

    for epoch in range(1, epochs + 1):
        train_loss, clean_acc, adv_acc = train_adversarial(
            defense_model, device, train_loader, optimizer, criterion, attack_fn, args.epsilon
        )
        test_loss, test_acc = evaluate(defense_model, device, test_loader, criterion)
        print(f'Epoch {epoch}: Loss: {train_loss:.4f} | Train Clean: {clean_acc:.2f}% | Train Adv: {adv_acc:.2f}% | Test Clean: {test_acc:.2f}%')

    torch.save(defense_model.state_dict(), save_path)
    print(f'\nModel saved to {save_path}')

if __name__ == '__main__':
    main()