# src/training/adversarial.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from src.attacks.registry import get_attack_fn
from src.datasets.mnist import get_train_loader, get_test_loader
from src.evaluation.core import evaluate
from src.models.factory import load_model, load_or_create_model
from src.utils.config import load_experiment, ExperimentConfig, TrainingConfig
from src.utils.logger import JSONLLogger
from src.utils.seed import set_seed, get_device


def train_epoch(model, device, loader, optimizer, criterion, attack_fn, epsilon):
    model.train()

    total_loss    = 0
    correct_clean = 0
    correct_adv   = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        adv_data = attack_fn(model, device, data, target, epsilon)

        combined_data   = torch.cat((data, adv_data), dim=0)
        combined_target = torch.cat((target, target), dim=0)

        optimizer.zero_grad()
        output = model(combined_data)
        loss   = criterion(output, combined_target)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        correct_clean += (output[:data.size(0)].argmax(dim=1) == target).sum().item()
        correct_adv   += (output[data.size(0):].argmax(dim=1) == target).sum().item()

    n = len(loader.dataset)
    return (
        total_loss / len(loader),
        100.0 * correct_clean / n,
        100.0 * correct_adv   / n,
    )


def _attack_tag(training_cfg: TrainingConfig) -> str:
    attack = training_cfg.attack
    if attack.steps is not None:
        return f"{attack.name}{attack.steps}"
    return attack.name


def train(cfg: ExperimentConfig, training_cfg: TrainingConfig, seed: int):
    set_seed(seed)
    device = get_device()

    tag             = _attack_tag(training_cfg)
    checkpoint_path = cfg.paths.checkpoints / f"adv_{tag}_seed{seed}.pth"
    base_path       = cfg.paths.checkpoints / f"standard_seed{seed}.pth"
    log_path        = cfg.paths.logs / f"adv_{tag}.jsonl"

    if checkpoint_path.exists():
        print(f"Skipping adv training {tag} seed={seed} — checkpoint exists")
        return

    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)

    logger       = JSONLLogger(str(log_path))
    train_loader = get_train_loader(cfg.dataset, batch_size=training_cfg.batch_size, seed=seed)
    test_loader  = get_test_loader(cfg.dataset,  batch_size=training_cfg.batch_size)

    # start from standard checkpoint if available, otherwise fresh model
    if base_path.exists():
        model = load_model(str(base_path), device, cfg.model)
        print(f"  Loaded standard checkpoint from {base_path}")
    else:
        model = load_or_create_model(cfg.model, device)
        print(f"  No standard checkpoint found, training from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)

    # resolve alpha — budget_scaled or fixed
    steps = training_cfg.attack.steps
    alpha = training_cfg.attack.alpha
    if alpha is None and steps is not None:
        alpha = 2.5 * training_cfg.epsilon / steps

    attack_fn, attack_params = get_attack_fn(
        training_cfg.attack.name,
        steps=steps,
        alpha=alpha,
    )

    print(f"  Adversarial training — {tag}, ε={training_cfg.epsilon}, seed={seed}")

    for epoch in range(training_cfg.epochs):
        train_loss, clean_acc, adv_acc = train_epoch(
            model, device, train_loader, optimizer, criterion,
            attack_fn, training_cfg.epsilon,
        )
        test_acc = evaluate(model, device, test_loader)

        print(f"  epoch {epoch+1}/{training_cfg.epochs}  "
              f"loss={train_loss:.4f}  clean={clean_acc:.1f}%  "
              f"adv={adv_acc:.1f}%  test={test_acc:.1f}%")

        logger.log({
            "run_type":             "adv_training",
            "dataset":              cfg.dataset.name,
            "model":                cfg.model.name,
            "seed":                 seed,
            "attack":               training_cfg.attack.name,
            "attack_params":        attack_params,
            "epsilon":              float(training_cfg.epsilon),
            "epoch":                epoch + 1,
            "train_loss":           float(train_loss),
            "train_clean_accuracy": float(clean_acc),
            "train_adv_accuracy":   float(adv_acc),
            "test_clean_accuracy":  float(test_acc),
        })

    torch.save(model.state_dict(), checkpoint_path)
    print(f"  Saved checkpoint → {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",    type=str,   required=True)
    parser.add_argument("--training-config", type=str, required=True,
                        help="Which training config to use, e.g. adv_pgd10")
    parser.add_argument("--seed",          type=int,   required=True)
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    cfg          = load_experiment(args.experiment, dry_run=args.dry_run)
    training_cfg = next(
        t for t in cfg.training
        if t.method == "adversarial" and _attack_tag(t) == args.training_config
    )

    train(cfg, training_cfg, seed=args.seed)


if __name__ == "__main__":
    main()