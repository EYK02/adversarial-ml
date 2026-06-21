# src/training/adversarial.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time

from src.attacks.registry import get_attack_fn
from src.datasets.mnist import get_train_loader, get_test_loader
from src.evaluation.core import evaluate
from src.models.factory import load_model, load_or_create_model
from src.utils.config import load_experiment, TrainingConfig
from src.utils.logger import JSONLLogger
from src.utils.seed import set_seed, get_device


def train_epoch(model, device, loader, optimizer, criterion, attack_fn, epsilon):
    model.train()

    total_loss = 0
    correct_clean = 0
    correct_adv = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        adv_data = attack_fn(model, device, data, target, epsilon)

        combined_data = torch.cat((data, adv_data), dim=0)
        combined_target = torch.cat((target, target), dim=0)

        optimizer.zero_grad()
        output = model(combined_data)
        loss = criterion(output, combined_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_clean += (output[: data.size(0)].argmax(dim=1) == target).sum().item()
        correct_adv += (output[data.size(0):].argmax(dim=1) == target).sum().item()

    n = len(loader.dataset)

    return (
        total_loss / len(loader),
        100.0 * correct_clean / n,
        100.0 * correct_adv / n,
    )


def _attack_tag(cfg: TrainingConfig) -> str:
    if cfg.attack.steps is not None:
        return f"{cfg.attack.name}{cfg.attack.steps}"
    return cfg.attack.name


def train(cfg, training_cfg, seed: int):
    set_seed(seed)
    device = get_device()

    tag = _attack_tag(training_cfg)

    # ── paths ─────────────────────────────
    run_ckpt_dir = cfg.paths.checkpoints / f"adv_{tag}_seed{seed}"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = run_ckpt_dir / "latest.pth"
    final_ckpt = run_ckpt_dir / "final.pth"
    best_ckpt = run_ckpt_dir / "best.pth"

    if final_ckpt.exists():
        print(f"[SKIP] {tag} seed={seed} already completed.")
        return

    log_path = cfg.paths.logs / f"adv_{tag}_seed{seed}.jsonl"
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)

    logger = JSONLLogger(str(log_path))

    train_loader = get_train_loader(cfg.dataset, training_cfg.batch_size, seed)
    test_loader = get_test_loader(cfg.dataset, training_cfg.batch_size)

    start_epoch = 0
    best_test_acc = 0.0

    if latest_ckpt.exists():
        print(f"[RESUME] loading {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, map_location=device)

        model = load_or_create_model(cfg.model, device)
        model.load_state_dict(ckpt["model"])

        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
        optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt["epoch"] + 1
        best_test_acc = ckpt.get("best_test_acc", 0.0)

    else:
        model = load_or_create_model(cfg.model, device)
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)

    criterion = nn.CrossEntropyLoss()

    # ── attack setup ───────────────────────
    steps = training_cfg.attack.steps
    alpha = training_cfg.attack.alpha

    if alpha is None and steps is not None:
        alpha = 2.5 * training_cfg.epsilon / steps

    attack_fn, _ = get_attack_fn(
        training_cfg.attack.name,
        steps=steps,
        alpha=alpha,
    )

    print(f"[TRAIN] {tag}, seed={seed}, eps={training_cfg.epsilon}")

    for epoch in range(start_epoch, training_cfg.epochs):

        start_time = time.perf_counter()

        train_loss, clean_acc, adv_acc = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            attack_fn,
            training_cfg.epsilon,
        )

        test_acc = evaluate(model, device, test_loader)

        duration = time.perf_counter() - start_time

        print(
            f"  epoch {epoch+1}/{training_cfg.epochs} | "
            f"loss={train_loss:.4f} | "
            f"clean={clean_acc:.1f}% | "
            f"adv={adv_acc:.1f}% | "
            f"test={test_acc:.1f}% | "
            f"duration={duration:.1f}s"
        )

        # ── logging ───────────────────────
        logger.log({
            "run_type": "adv_training",
            "dataset": cfg.dataset.name,
            "model": cfg.model.name,
            "seed": seed,
            "attack": training_cfg.attack.name,
            "epsilon": float(training_cfg.epsilon),
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_clean_accuracy": float(clean_acc),
            "train_adv_accuracy": float(adv_acc),
            "test_clean_accuracy": float(test_acc),
            "duration_sec": duration
        })

        # ── best checkpoint ───────────────
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_ckpt)

        # ── lightweight checkpoint (compressed) ─
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_test_acc": best_test_acc,
            },
            latest_ckpt,
            _use_new_zipfile_serialization=True,
        )

    # ── final model ───────────────────────
    torch.save(model.state_dict(), final_ckpt, _use_new_zipfile_serialization=True)

    print(f"[DONE] saved → {final_ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--training-config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    cfg = load_experiment(args.experiment, dry_run=args.dry_run, smoke_test=args.smoke_test)

    training_cfg = next(
        t for t in cfg.training
        if t.method == "adversarial" and _attack_tag(t) == args.training_config
    )

    train(cfg, training_cfg, seed=args.seed)


if __name__ == "__main__":
    main()