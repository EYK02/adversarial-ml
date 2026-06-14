from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_test_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_mnist_train_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)