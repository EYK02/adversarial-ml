# defenses/evaluate_defense.py

import argparse
import time
import torch
from src.models.cnn import CNN
from src.attacks.registry import get_attack_fn
from src.utils.data import get_mnist_test_loader
from src.utils.config import EPSILONS
from src.utils.reproducibility import set_seed
from src.evaluation.core import evaluate
from src.logging.logger import JSONLLogger
from src.logging.run_id import make_run_id

batch_size = 64

logger = JSONLLogger("results/jsonl/defense_eval.jsonl")


def _defense_model_path(attack: str, steps: int | None, epsilon: float, seed: int) -> str:
    if attack == "pgd" and steps is not None:
        attack_tag = f"pgd{steps}"
    else:
        attack_tag = attack
    return f"models/cnn_mnist_adv_{attack_tag}_eps{epsilon}_seed{seed}.pth"


def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial defense on MNIST")
    parser.add_argument("--defense_attack",  type=str,   required=True, help="Attack used during adversarial training (fgsm, pgd)")
    parser.add_argument("--defense_steps",   type=int,   default=None,  help="PGD steps used during adversarial training (PGD only)")
    parser.add_argument("--defense_epsilon", type=float, required=True, help="Epsilon used during adversarial training")
    parser.add_argument("--eval_attack",     type=str,   required=True, help="Attack to evaluate against (fgsm, pgd)")
    parser.add_argument("--eval_steps",      type=int,   default=None,  help="PGD steps for evaluation attack (PGD only)")
    parser.add_argument("--seed",            type=int,   default=0,     help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    run_id = make_run_id(
        task="defense_eval",
        model="cnn_mnist",
        defense=args.defense_attack,
        defense_steps=args.defense_steps,
        eval_attack=args.eval_attack,
        eval_steps=args.eval_steps,
        defense_epsilon=args.defense_epsilon,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_mnist_test_loader(batch_size)

    # baseline model
    base_model_path = f"models/cnn_mnist_seed{args.seed}.pth"
    base_model = CNN().to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    base_model.eval()

    # defense model
    defense_path = _defense_model_path(
        args.defense_attack, args.defense_steps, args.defense_epsilon, args.seed
    )
    defense_model = CNN().to(device)
    defense_model.load_state_dict(torch.load(defense_path, map_location=device))
    defense_model.eval()

    # eval attack
    eval_attack_fn, eval_attack_params = get_attack_fn(args.eval_attack, steps=args.eval_steps)

    # defense identity for logging
    if args.defense_attack == "pgd" and args.defense_steps is not None:
        defense_tag = f"pgd{args.defense_steps}"
    else:
        defense_tag = args.defense_attack

    print(f"Defense eval — defense={defense_tag}, eval_attack={args.eval_attack}, seed={args.seed}\n")

    for epsilon in EPSILONS:
        start = time.perf_counter()

        base_acc    = evaluate(base_model,    device, test_loader, eval_attack_fn, epsilon)
        defense_acc = evaluate(defense_model, device, test_loader, eval_attack_fn, epsilon)

        duration = time.perf_counter() - start

        print(
            f"  eps={epsilon:.2f}  base={base_acc:.2f}%  defense={defense_acc:.2f}%  "
            f"delta={defense_acc - base_acc:+.2f}%"
        )

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