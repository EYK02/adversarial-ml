import torch

def evaluate(ctx):
    ctx.model.eval()

    loader = ctx.loaders["test"]

    total_loss = 0.0
    correct = 0
    total = 0

    attack_fn = getattr(ctx, "attack_fn", None)
    epsilon = getattr(ctx, "epsilon", None)

    use_loss = ctx.criterion is not None

    for x, y in loader:
        x = x.to(ctx.device)
        y = y.to(ctx.device)

        if attack_fn is not None:
            x = attack_fn(
                ctx.model,
                ctx.device,
                x,
                y,
                epsilon,
            )

        with torch.no_grad():
            out = ctx.model(x)

            if use_loss:
                loss = ctx.criterion(out, y)
                total_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total

    if use_loss:
        return total_loss / len(loader), acc
    return acc
