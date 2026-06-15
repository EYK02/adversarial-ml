# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from utils.data import get_mnist_train_loader, get_mnist_test_loader
from utils.reproducibility import set_seed
from utils.evaluation import evaluate

batch_size = 64
learning_rate = 0.001
num_epochs = 5

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy

def main():
    os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_mnist_train_loader(batch_size, seed=args.seed)
    test_loader = get_mnist_test_loader(batch_size)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion=criterion)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    model_save_path = f'models/cnn_mnist_seed{args.seed}.pth'
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()