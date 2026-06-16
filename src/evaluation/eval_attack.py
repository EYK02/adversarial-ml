# attacks/evaluate_attack.py

import argparse
import time
import torch
from src.models.factory import load_model
from src.attacks.registry import get_attack_fn
from src.utils.config import EPSILONS
from src.data.loader import get_mnist_test_loader
from src.evaluation.core import evaluate
from src.logging.logger import JSONLLogger
from src.utils.reproducibility import set_seed
from src.logging.run_id import make_run_id

batch_size = 64

logger = JSONLLogger("results/jsonl/attack_eval.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial attack on MNIST")
    parser.add_argument("--attack", type=str,   default="fgsm", help="Attack to evaluate")
    parser.add_argument("--steps",  type=int,   default=None,   help="PGD step count (PGD only)")
    parser.add_argument("--seed",   type=int,   default=0,      help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    base_model_path = f"models/cnn_mnist_seed{args.seed}.pth"

    run_id = make_run_id(
        task="attack_eval",
        model="cnn_mnist",
        attack=args.attack,
        epsilon=None,
        seed=args.seed,
    )

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(base_model_path, device)


    attack_fn, attack_params = get_attack_fn(args.attack, steps=args.steps)
    test_loader = get_mnist_test_loader(batch_size)

    print(f"Attack eval — attack={args.attack}, params={attack_params}, seed={args.seed}\n")

    for epsilon in EPSILONS:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        acc = evaluate(model, device, test_loader, attack_fn, epsilon)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start

        print(f"  eps={epsilon:.2f}  acc={acc:.2f}%  ({duration:.1f}s)")

        logger.log({
            "run_id":        run_id,
            "run_type":      "attack_eval",
            "model":         "cnn_mnist",
            "model_path":    base_model_path,
            "dataset":       "mnist",
            "seed":          args.seed,

            "attack":        args.attack,
            "attack_params": attack_params,
            "epsilon":       float(epsilon),

            "metric":        "accuracy",
            "value":         float(acc),
            "duration_sec":  float(duration),
        })


if __name__ == "__main__":
    main()