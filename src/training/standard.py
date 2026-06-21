# src/training/standard.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.mnist import get_train_loader, get_test_loader
from src.evaluation.core import evaluate
from src.models.factory import load_or_create_model
from src.utils.config import load_experiment, ExperimentConfig, TrainingConfig
from src.utils.logger import JSONLLogger
from src.utils.seed import set_seed, get_device


def train_epoch(model, device, loader, optimizer, criterion):
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
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / len(loader.dataset)


def train(cfg: ExperimentConfig, training_cfg: TrainingConfig, seed: int):
    set_seed(seed)
    device = get_device()

    checkpoint_path = cfg.paths.checkpoints / f"standard_seed{seed}.pth"
    log_path        = cfg.paths.logs / "standard.jsonl"

    if checkpoint_path.exists():
        print(f"Skipping standard training seed={seed} — checkpoint exists")
        return

    logger       = JSONLLogger(str(log_path))
    train_loader = get_train_loader(cfg.dataset, batch_size=training_cfg.batch_size, seed=seed)
    test_loader  = get_test_loader(cfg.dataset,  batch_size=training_cfg.batch_size)

    model     = load_or_create_model(cfg.model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)

    for epoch in range(training_cfg.epochs):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss,  test_acc  = evaluate(model, device, test_loader, criterion=criterion)

        print(f"  epoch {epoch+1}/{training_cfg.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%")

        logger.log({
            "run_type":       "training",
            "dataset":        cfg.dataset.name,
            "model":          cfg.model.name,
            "seed":           seed,
            "epoch":          epoch + 1,
            "train_loss":     float(train_loss),
            "train_accuracy": float(train_acc),
            "test_loss":      float(test_loss),
            "test_accuracy":  float(test_acc),
        })

    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  Saved checkpoint → {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--seed",       type=int, required=True)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    cfg          = load_experiment(args.experiment, dry_run=args.dry_run)
    training_cfg = next(t for t in cfg.training if t.method == "standard")

    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)

    train(cfg, training_cfg, seed=args.seed)


if __name__ == "__main__":
    main()