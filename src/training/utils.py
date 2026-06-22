import torch


def is_training_complete(ctx) -> bool:
    return (
        ctx.final_ckpt.exists() 
        and ctx.best_ckpt.exists()
        and ctx.latest_ckpt.exists()
    )


def get_inputs(ctx, x, y):
    if ctx.attack_fn is None:
        return x, y

    x_adv = ctx.attack_fn(
        ctx.model,
        ctx.device,
        x,
        y,
        ctx.epsilon,
    )

    x = torch.cat((x, x_adv), dim=0)
    y = torch.cat((y, y), dim=0)

    return x, y


def train_epoch(ctx):
    ctx.model.train()

    loader = ctx.loaders["train"]

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(ctx.device, non_blocking=True)
        y = y.to(ctx.device, non_blocking=True)

        ctx.optimizer.zero_grad()

        x, y = get_inputs(ctx, x, y)

        out = ctx.model(x)
        loss = ctx.criterion(out, y)

        loss.backward()
        ctx.optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), 100.0 * correct / total