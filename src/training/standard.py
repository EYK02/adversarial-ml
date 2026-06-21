# src/training/standard.py

import argparse
import time

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
    correct    = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / len(loader.dataset)


def train(cfg: ExperimentConfig, training_cfg: TrainingConfig, seed: int):
    set_seed(seed)
    device = get_device()

    # ── paths ─────────────────────────────────────────────────────
    run_ckpt_dir = cfg.paths.checkpoints / f"standard_seed{seed}"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = run_ckpt_dir / "latest.pth"
    final_ckpt  = run_ckpt_dir / "final.pth"
    best_ckpt   = run_ckpt_dir / "best.pth"

    if final_ckpt.exists():
        print(f"[SKIP] standard seed={seed} already completed.")
        return

    log_path = cfg.paths.logs / "standard.jsonl"
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)

    logger       = JSONLLogger(str(log_path))
    train_loader = get_train_loader(cfg.dataset, training_cfg.batch_size, seed)
    test_loader  = get_test_loader(cfg.dataset,  training_cfg.batch_size)

    criterion    = nn.CrossEntropyLoss()
    start_epoch  = 0
    best_test_acc = 0.0

    # ── resume from latest checkpoint if available ─────────────────
    if latest_ckpt.exists():
        print(f"[RESUME] loading {latest_ckpt}")
        ckpt  = torch.load(latest_ckpt, map_location=device)
        model = load_or_create_model(cfg.model, device)
        model.load_state_dict(ckpt["model"])
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_test_acc = ckpt.get("best_test_acc", 0.0)
    else:
        model     = load_or_create_model(cfg.model, device)
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)

    print(f"[TRAIN] standard, seed={seed}")

    for epoch in range(start_epoch, training_cfg.epochs):
        start_time = time.perf_counter()

        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss,  test_acc  = evaluate(model, device, test_loader, criterion=criterion)

        duration = time.perf_counter() - start_time

        print(
            f"  epoch {epoch+1}/{training_cfg.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train={train_acc:.1f}% | "
            f"test={test_acc:.1f}% | "
            f"{duration:.1f}s"
        )

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
            "duration":       float(duration),
        })

        # ── best checkpoint ────────────────────────────────────────
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_ckpt)

        # ── latest checkpoint for resume ───────────────────────────
        torch.save(
            {
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "epoch":         epoch,
                "best_test_acc": best_test_acc,
            },
            latest_ckpt,
            _use_new_zipfile_serialization=True,
        )

    # ── final model ────────────────────────────────────────────────
    torch.save(model.state_dict(), final_ckpt, _use_new_zipfile_serialization=True)
    print(f"[DONE] saved → {final_ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--seed",       type=int, required=True)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    cfg          = load_experiment(args.experiment, dry_run=args.dry_run)
    training_cfg = next(t for t in cfg.training if t.method == "standard")

    cfg.paths.logs.mkdir(parents=True,        exist_ok=True)
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)

    train(cfg, training_cfg, seed=args.seed)


if __name__ == "__main__":
    main()