# defenses/evaluate_defense.py

import argparse
import torch
from model import CNN
from utils.data import get_mnist_test_loader
from utils.config import EPSILONS
from utils.reproducibility import set_seed
from utils.evaluation import evaluate
from utils.logger import JSONLLogger
from utils.metrics import compute_attack_metrics

batch_size = 64

logger = JSONLLogger("results/jsonl/defense_eval.jsonl")

def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial defense on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm',
                        help='Attack to evaluate against')
    parser.add_argument('--defense', type=str, default='fgsm',
                        help='Defended model to evaluate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model_path = f'models/cnn_mnist_seed{args.seed}.pth'

    base_model = CNN().to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    base_model.eval()

    defense_model = CNN().to(device)
    defense_model.load_state_dict(torch.load(MODELS[args.defense], map_location=device))
    defense_model.eval()

    attack_fn = ATTACKS[args.attack]
    test_loader = get_mnist_test_loader(batch_size)

    for epsilon in EPSILONS:
        base_acc = evaluate(base_model, device, test_loader, attack_fn, epsilon)
        def_acc = evaluate(defense_model, device, test_loader, attack_fn, epsilon)
        
        metrics = compute_attack_metrics(epsilon, base_acc, def_acc)
        delta = metrics["delta"]
        attack_success = metrics["attack_success"]

        logger.log({
            "run_type":     "defense_eval",
            "dataset":      "mnist",

            "attack":       args.attack,
            "defense":      args.defense,
            "seed":         0,
            "epsilon":      float(epsilon),

            "baseline_accuracy":    float(base_acc),
            "defended_Accuracy":    float(def_acc),
            "delta":                float(delta),
            "successful_attack":    float(attack_success)
        })


if __name__ == '__main__':
    main()