# utils/evaluation.py

import torch

def evaluate(model, device, loader, attack_fn=None, epsilon=None, criterion=None):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    use_loss = criterion is not None
    use_attack = attack_fn is not None

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if use_attack:
            x = attack_fn(model, device, x, y, epsilon)

        with torch.no_grad():
            out = model(x)

            if use_loss:
                loss = criterion(out, y)
                total_loss += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total

    if use_loss:
        return total_loss / len(loader), acc
    return acc

def attack_success_rate(model, device, loader, attack_fn, epsilon):
    acc = evaluate(model, device, loader, attack_fn, epsilon)
    return (1.0 - acc / 100.0) * 100