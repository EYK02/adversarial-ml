import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN
from attacks.fgsm import fgsm_attack

base_model_path = 'models/cnn_mnist.pth'
defense_model_path = 'models/cnn_mnist_fgsm_adv.pth'
batch_size = 64
epochs = 5
learning_rate = 0.001
defense_epsilon = 0.2

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_adversarial(model, device, train_loader, optimizer, criterion, epsilon):
    model.train()
    
    total_loss = 0
    correct_clean = 0
    correct_adv = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        adv_data = fgsm_attack(model, device, data, target, epsilon)
        
        # Combine clean and adversarial examples
        combined_data = torch.cat((data, adv_data), dim=0)
        combined_target = torch.cat((target, target), dim=0)
        
        optimizer.zero_grad()
        output = model(combined_data)
        loss = criterion(output, combined_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy on clean and adversarial examples
        _, predicted_clean = torch.max(output[:data.size(0)].data, 1)
        _, predicted_adv = torch.max(output[data.size(0):].data, 1)
        
        correct_clean += (predicted_clean == target).sum().item()
        correct_adv += (predicted_adv == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    clean_acc = 100 * correct_clean / (len(train_loader.dataset))
    adv_acc = 100 * correct_adv / (len(train_loader.dataset))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Initialize defense model
    defense_model = CNN().to(device)
    defense_model.load_state_dict(torch.load(base_model_path, map_location=device))   
    
    optimizer = optim.Adam(defense_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Adversarial training
    for epoch in range(1, epochs + 1):
        train_loss, clean_acc, adv_acc = train_adversarial(defense_model, device, train_loader, optimizer, criterion, defense_epsilon)
        test_loss, test_acc = evaluate(defense_model, device, test_loader, criterion)
        print(f'Epoch {epoch}: Loss: {train_loss:.4f} | Train Clean: {clean_acc:.2f}% | Train Adv: {adv_acc:.2f}% | Test Clean: {test_acc:.2f}%')
    
    # Save the adversarially trained model
    torch.save(defense_model.state_dict(), defense_model_path)

if __name__ == "__main__":
    main()