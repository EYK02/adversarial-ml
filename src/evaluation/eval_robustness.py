# src/evaluation/eval_robustness.py

import argparse
import time
from src.attacks.registry import get_attack_fn
from src.data.loader import get_mnist_test_loader
from src.evaluation.core import evaluate
from src.logging.logger import JSONLLogger
from src.logging.run_id import make_run_id
from src.models.factory import load_model
from src.utils.config import EPSILONS, BATCH_SIZE
from src.utils.reproducibility import set_seed, get_device

logger = JSONLLogger("artifacts/jsonl/defense_eval.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial defense on MNIST")
    parser.add_argument("--defense_attack",  type=str,   required=True, help="Attack used during adversarial training (fgsm, pgd)")
    parser.add_argument("--defense_steps",   type=int,   default=None,  help="PGD steps used during adversarial training (PGD only)")
    parser.add_argument("--defense_epsilon", type=float, required=True, help="Epsilon used during adversarial training")
    parser.add_argument("--eval_attack",     type=str,   required=True, help="Attack to evaluate against (fgsm, pgd)")
    parser.add_argument("--eval_steps",      type=int,   default=None,  help="PGD steps for evaluation attack (PGD only)")
    parser.add_argument("--seed",            type=int,   default=0,     help="Random seed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        from src.utils.config import EPSILONS_DRY as EPSILONS, DEFENSES_DRY as DEFENSES, EVAL_ATTACKS_DRY as EVAL_ATTACKS, NUM_SEEDS_DRY as SEEDS
    else:
        from src.utils.config import EPSILONS as EPSILONS, DEFENSES as DEFENSES, EVAL_ATTACKS as EVAL_ATTACKS, NUM_SEEDS as SEEDS

    set_seed(args.seed)
    device = get_device()
    test_loader = get_mnist_test_loader(BATCH_SIZE)

    # baseline model
    base_model_path = f"models/cnn_mnist_seed{args.seed}.pth"
    base_model      = load_model(base_model_path, device)

    # defense model
    defense_tag      = f'pgd_steps{args.defense_steps}' if args.defense_attack == "pgd" and args.defense_steps is not None else args.defense_attack
    defense_path    = f"models/adv_train_cnn_mnist_attack{defense_tag}_eps{args.defense_epsilon}_seed{args.seed}.pth"
    defense_model   = load_model(defense_path, device)

    # eval attack
    eval_attack_fn, eval_attack_params = get_attack_fn(args.eval_attack, steps=args.eval_steps)

    print(f"Defense eval — defense={defense_tag}, eval_attack={args.eval_attack}, seed={args.seed}\n")

    for epsilon in EPSILONS:
        # Generate run_id
        run_id = make_run_id(
            task="defense_eval",
            model="cnn_mnist",
            defense=args.defense_attack,
            defense_steps=args.defense_steps,
            eval_attack=args.eval_attack,
            eval_steps=args.eval_steps,
            defense_epsilon=args.defense_epsilon,
            epsilon=epsilon,
            seed=args.seed,
        )

        # Skip if evaluation already exists
        if logger.contains(run_id):
            print(f"   Skipping eps={epsilon:.2f} (already completed)")
            continue

        # Evaluate (timed)
        start = time.perf_counter()

        base_acc    = evaluate(base_model,    device, test_loader, eval_attack_fn, epsilon)
        defense_acc = evaluate(defense_model, device, test_loader, eval_attack_fn, epsilon)

        duration = time.perf_counter() - start

        print(f"  eps={epsilon:.2f}  base={base_acc:.2f}%  defense={defense_acc:.2f}%  delta={defense_acc - base_acc:+.2f}%")

        # Log evaluation
        logger.log({
            "run_id":   run_id,
            "run_type": "defense_eval",
            "model":    "cnn_mnist",
            "dataset":  "mnist",
            "seed":     args.seed,

            "defense_attack":  args.defense_attack,
            "defense_params":  {"steps": args.defense_steps} if args.defense_steps else {},
            "defense_epsilon": args.defense_epsilon,
            "defense_path":    defense_path,

            "eval_attack":  args.eval_attack,
            "eval_params":  eval_attack_params,
            "epsilon":      float(epsilon),

            "baseline_accuracy": float(base_acc),
            "defense_accuracy":  float(defense_acc),
            "delta":             float(defense_acc - base_acc),
            "duration_sec":      float(duration),
        })


if __name__ == "__main__":
    main()