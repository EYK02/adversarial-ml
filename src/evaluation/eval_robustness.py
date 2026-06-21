# src/evaluation/eval_robustness.py

import argparse
import time

from src.attacks.registry import get_attack_fn
from src.datasets.mnist import get_test_loader
from src.evaluation.core import evaluate
from src.models.factory import load_model
from src.utils.config import load_experiment, ExperimentConfig, AttackConfig, TrainingConfig
from src.utils.logger import JSONLLogger
from src.utils.run_id import make_run_id
from src.utils.seed import set_seed, get_device


def _attack_tag(training_cfg: TrainingConfig) -> str:
    attack = training_cfg.attack
    if attack.steps is not None:
        return f"{attack.name}{attack.steps}"
    return attack.name


def eval_robustness(
        cfg:          ExperimentConfig,
        training_cfg: TrainingConfig,
        eval_cfg:     AttackConfig,
        seed:         int,
        epsilon:      float,
        logger:       JSONLLogger,
) -> None:
    device      = get_device()
    test_loader = get_test_loader(cfg.dataset, batch_size=64)

    # load baseline and defense checkpoints
    base_path    = cfg.paths.checkpoints / f"standard_seed{seed}" / "final.pth"
    defense_tag  = _attack_tag(training_cfg)
    defense_path = cfg.paths.checkpoints / f"adv_{defense_tag}_seed{seed}" / "final.pth"

    base_model    = load_model(str(base_path),    device, cfg.model)
    defense_model = load_model(str(defense_path), device, cfg.model)

    # resolve eval attack alpha
    eval_steps = eval_cfg.steps
    eval_alpha = eval_cfg.alpha
    if eval_alpha is None and eval_steps is not None:
        eval_alpha = 2.5 * epsilon / eval_steps

    eval_attack_fn, eval_attack_params = get_attack_fn(
        eval_cfg.name,
        steps=eval_steps,
        alpha=eval_alpha,
        restarts=eval_cfg.restarts
    )

    run_id = make_run_id(
        task          = "defense_eval",
        model         = cfg.model.name,
        dataset       = cfg.dataset.name,
        defense       = training_cfg.attack.name,
        defense_steps = training_cfg.attack.steps,
        eval_attack   = eval_cfg.name,
        eval_steps    = eval_steps,
        epsilon       = epsilon,
        seed          = seed,
    )

    if logger.contains(run_id):
        print(f"  Skipping {run_id} — already logged")
        return

    start        = time.perf_counter()
    base_acc     = evaluate(base_model,    device, test_loader, eval_attack_fn, epsilon)
    defense_acc  = evaluate(defense_model, device, test_loader, eval_attack_fn, epsilon)
    duration     = time.perf_counter() - start

    print(
        f"  eps={epsilon:.2f} | "
        f"base={base_acc:.2f}% | "
        f"defense={defense_acc:.2f}% | "
        f"delta={defense_acc - base_acc:+.2f}% | "
        f"{duration:.1f}s"
    )

    logger.log({
        "run_id":            run_id,
        "run_type":          "defense_eval",
        "model":             cfg.model.name,
        "dataset":           cfg.dataset.name,
        "seed":              seed,

        "defense_attack":    training_cfg.attack.name,
        "defense_params":    {"steps": training_cfg.attack.steps}
                             if training_cfg.attack.steps else {},
        "defense_epsilon":   float(training_cfg.epsilon),
        "defense_path":      str(defense_path),

        "eval_attack":       eval_cfg.name,
        "eval_params":       eval_attack_params,
        "epsilon":           float(epsilon),

        "baseline_accuracy": float(base_acc),
        "defense_accuracy":  float(defense_acc),
        "delta":             float(defense_acc - base_acc),
        "duration_sec":      float(duration),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",      type=str, required=True)
    parser.add_argument("--training-config", type=str, required=True,
                        help="Which adversarial training config, e.g. fgsm or pgd10")
    parser.add_argument("--eval-attack",     type=str, required=True,
                        help="Eval attack name, e.g. fgsm or pgd40")
    parser.add_argument("--seed",            type=int, required=True)
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--smoke-test",    action="store_true")
    args = parser.parse_args()

    cfg = load_experiment(args.experiment, dry_run=args.dry_run, smoke_test=args.smoke_test)

    training_cfg = next(
        t for t in cfg.training
        if t.method == "adversarial" and _attack_tag(t) == args.training_config
    )

    eval_cfg = next(
        a for a in cfg.eval_attacks
        if (a.name == args.eval_attack)
        or (a.steps is not None and f"{a.name}{a.steps}" == args.eval_attack)
    )

    log_path = cfg.paths.logs / "defense_eval.jsonl"
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(str(log_path))

    set_seed(args.seed)
    print(f"Defense eval — defense={args.training_config}, "
          f"eval={args.eval_attack}, seed={args.seed}")

    for epsilon in cfg.epsilon_eval:
        eval_robustness(cfg, training_cfg, eval_cfg, args.seed, epsilon, logger)


if __name__ == "__main__":
    main()