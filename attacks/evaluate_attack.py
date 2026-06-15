# attacks/evaluate_attack.py

import argparse
import torch
from attacks.registry import ATTACKS
from utils.data import get_mnist_test_loader
from utils.config import EPSILONS
from utils.reproducibility import set_seed
from utils.evaluation import evaluate
from utils.logger import JSONLLogger
from utils.modeling import load_model
batch_size = 64

logger = JSONLLogger("results/jsonl/attack_eval.jsonl")

def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial attack on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm', choices=ATTACKS.keys(), help='Attack to evaluate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    set_seed(args.seed)
    base_model_path = f'models/cnn_mnist_seed{args.seed}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(base_model_path, device)

    attack_fn = ATTACKS[args.attack]
    test_loader = get_mnist_test_loader(batch_size)

    for epsilon in EPSILONS:
        acc = evaluate(model, device, test_loader, attack_fn, epsilon)
        logger.log({
            "run_type":     "attack_eval",
            "model":        "cnn_mnist",
            "model_path":   base_model_path,
            "dataset":      "mnist",
            "attack":       args.attack,
            "epsilon":      float(epsilon),
            "metric":       "accuracy",
            "value":        float(acc),
            "seed":         args.seed
        })

if __name__ == '__main__':
    main()