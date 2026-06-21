# src/training/standard.py

import argparse
import time

import torch

from src.evaluation.core import evaluate
from src.utils.config import load_experiment
from src.utils.context import RunContext, build_train_ctx


def train_epoch(ctx: RunContext):
    ctx.model.train()

    loader = ctx.loaders["train"]
    total_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(ctx.device), labels.to(ctx.device)

        ctx.optimizer.zero_grad()
        outputs = ctx.model(images)
        loss = ctx.criterion(outputs, labels)

        loss.backward()
        ctx.optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    return (
        total_loss / len(loader),
        100.0 * correct / len(loader.dataset)
    )


def train(ctx: RunContext):
    if ctx is None:
        print(f"[SKIP] {training_cfg.method} seed={seed} already completed.")
        return

    print(f"[TRAIN] {ctx.training_cfg.method}, seed={ctx.seed}")
    
    start_epoch = 0

    for epoch in range(ctx.epoch, ctx.training_cfg.epochs):
        start = time.perf_counter()

        train_loss, train_acc = train_epoch(ctx)
        test_loss, test_acc = evaluate(ctx)

        duration = time.perf_counter() - start

        print(
            f"  epoch {epoch+1}/{ctx.training_cfg.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train={train_acc:.1f}% | "
            f"test={test_acc:.1f}% | "
            f"duration={duration:.1f}s"
        )

        ctx.logger.log({
            "run_type": "training",
            "dataset": ctx.cfg.dataset.name,
            "model": ctx.cfg.model.name,
            "seed": ctx.seed,
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "duration": float(duration),
        })

        # ── best checkpoint ───────────────────────────────
        if test_acc > ctx.best_acc:
            ctx.best_acc = test_acc
            torch.save(ctx.model.state_dict(), ctx.best_ckpt)

        # ── latest checkpoint ─────────────────────────────
        torch.save(
            {
                "model": ctx.model.state_dict(),
                "optimizer": ctx.optimizer.state_dict(),
                "epoch": epoch,
                "best_test_acc": ctx.best_acc,
            },
            ctx.latest_ckpt,
            _use_new_zipfile_serialization=True,
        )

    # ── final model ───────────────────────────────────────
    torch.save(
        ctx.model.state_dict(),
        ctx.final_ckpt,
        _use_new_zipfile_serialization=True,
    )

    print(f"[DONE] saved → {ctx.final_ckpt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    cfg = load_experiment(
        args.experiment,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        run_name=args.run_name
    )

    training_cfg = next(t for t in cfg.training if t.method == "standard")

    ctx = build_train_ctx(cfg, training_cfg, args.seed)

    train(ctx)


if __name__ == "__main__":
    main()