import torch

def evaluate(
    ctx,
    attack_fn=None,
    split: str = "test",
    epsilon=None,
):
    ctx.model.eval()

    loader = ctx.loaders[split]

    total_loss = 0.0
    correct = 0
    total = 0

    use_loss = ctx.criterion is not None
    use_attack = attack_fn is not None

    for x, y in loader:
        x, y = x.to(ctx.device), y.to(ctx.device)

        if use_attack:
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
                loss        = ctx.criterion(out, y)
                total_loss += loss.item()

            pred        = out.argmax(dim=1)
            correct    += (pred == y).sum().item()
            total      += y.size(0)

    acc = 100.0 * correct / total

    if use_loss:
        return total_loss / len(loader), acc
    return acc