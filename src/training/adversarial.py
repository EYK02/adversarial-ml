# src/training/adversarial.py

import argparse
import time
import torch
import sys

from src.runner.builders import build_adv_train_ctx
from src.runner.utils import attack_tag
from src.utils.config import load_experiment
from src.training.utils import train_epoch, is_training_complete
from src.evaluation.utils import evaluate


def train(ctx):
    # early skip
    if is_training_complete(ctx):
        print(f"[SKIP] adversarial {ctx.training_cfg.attack.name} seed={ctx.seed} already completed.")
        sys.exit(2)
        
    print(
        f"[TRAIN] adversarial {ctx.training_cfg.attack.name}, "
        f"seed={ctx.seed}, eps={ctx.training_cfg.epsilon}"
    )

    for epoch in range(ctx.epoch, ctx.training_cfg.epochs):

        start = time.perf_counter()

        train_loss, train_acc = train_epoch(ctx)
        test_loss, test_acc = evaluate(ctx)

        duration = time.perf_counter() - start

        print(
            f"  epoch {epoch+1}/{ctx.training_cfg.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.1f}% | "
            f"test={test_acc:.1f}% | "
            f"duration={duration:.1f}s"
        )

        ctx.logger.log({
            "run_type": "adv_training",
            "dataset": ctx.cfg.dataset.name,
            "model": ctx.cfg.model.name,
            "seed": ctx.seed,
            "attack": ctx.training_cfg.attack.name,
            "epsilon": float(ctx.training_cfg.epsilon),
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "test_clean_accuracy": float(test_acc),
            "duration_sec": duration,
        })

        # update best
        if test_acc > ctx.best_acc:
            ctx.best_acc = test_acc
            torch.save(ctx.model.state_dict(), ctx.best_ckpt)

        # checkpoint (resume-ready)
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

    # final model
    torch.save(ctx.model.state_dict(), ctx.final_ckpt)
    print(f"[DONE] saved → {ctx.final_ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--training-config", type=str, required=True)
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

    training_cfg = next(
        t for t in cfg.training
        if t.method == "adversarial"
        and attack_tag(t) == args.training_config
    )

    ctx = build_adv_train_ctx(cfg, training_cfg, args.seed)

    train(ctx)


if __name__ == "__main__":
    main()