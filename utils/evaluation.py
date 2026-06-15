import torch

def evaluate_clean(model, device, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total

def evaluate_adversarial(model, device, loader, attack_fn, epsilon):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        x_adv = attack_fn(model, device, x, y, epsilon)

        out = model(x_adv)
        pred = out.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total

def attack_success_rate(model, device, loader, attack_fn, epsilon):
    acc = evaluate_adversarial(model, device, loader, attack_fn, epsilon)
    return 100.0 - acc