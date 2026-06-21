# src/evaluation/eval_attack.py

import argparse
import time

from src.attacks.registry import get_attack_fn
from src.datasets.mnist import get_test_loader
from src.evaluation.core import evaluate
from src.models.factory import load_model
from src.utils.config import load_experiment, ExperimentConfig, AttackConfig
from src.utils.logger import JSONLLogger
from src.utils.run_id import make_run_id
from src.utils.seed import set_seed, get_device


def eval_attack(
        cfg:        ExperimentConfig,
        attack_cfg: AttackConfig,
        seed:       int,
        epsilon:    float,
        logger:     JSONLLogger,
) -> None:
    device      = get_device()
    test_loader = get_test_loader(cfg.dataset, batch_size=64)

    checkpoint_path = cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"
    model           = load_model(str(checkpoint_path), device, cfg.model)

    # resolve alpha
    steps = attack_cfg.steps
    alpha = attack_cfg.alpha
    if alpha is None and steps is not None:
        alpha = 2.5 * epsilon / steps

    attack_fn, attack_params = get_attack_fn(
        attack_cfg.name, 
        steps=steps, 
        alpha=alpha,
        restarts=attack_cfg.restarts
    )

    run_id = make_run_id(
        task    = "attack_eval",
        model   = cfg.model.name,
        dataset = cfg.dataset.name,
        attack  = attack_cfg.name,
        steps   = steps,
        epsilon = epsilon,
        seed    = seed,
    )

    if logger.contains(run_id):
        print(f"  Skipping {run_id} — already logged")
        return

    start = time.perf_counter()
    acc   = evaluate(model, device, test_loader, attack_fn, epsilon)
    duration = time.perf_counter() - start    
    
    print(
        f"  eps={epsilon:.2f} | "
        f"acc={acc:.2f}% | "
        f"{duration:.1f}s"
    )

    logger.log({
        "run_id":        run_id,
        "run_type":      "attack_eval",
        "model":         cfg.model.name,
        "model_path":    str(checkpoint_path),
        "dataset":       cfg.dataset.name,
        "seed":          seed,
        "attack":        attack_cfg.name,
        "attack_params": attack_params,
        "epsilon":       float(epsilon),
        "accuracy":      float(acc),
        "duration_sec":  float(duration),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--attack",     type=str, required=True,
                        help="Attack name, e.g. fgsm or pgd10")
    parser.add_argument("--seed",       type=int, required=True)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--smoke-test",    action="store_true")
    args = parser.parse_args()

    cfg = load_experiment(args.experiment, dry_run=args.dry_run, smoke_test=args.smoke_test)

    attack_cfg = next(
        a for a in cfg.eval_attacks
        if (a.name == args.attack)
        or (a.steps is not None and f"{a.name}{a.steps}" == args.attack)
    )

    log_path = cfg.paths.logs / "attack_eval.jsonl"
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(str(log_path))

    set_seed(args.seed)
    print(f"Attack eval — {args.attack}, seed={args.seed}")

    for epsilon in cfg.epsilon_eval:
        eval_attack(cfg, attack_cfg, args.seed, epsilon, logger)


if __name__ == "__main__":
    main()