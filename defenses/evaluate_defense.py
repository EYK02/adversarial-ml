# defenses/evaluate_defense.py

import argparse
import torch
from model import CNN
from attacks.registry import ATTACKS, MODELS
from utils.data import get_mnist_test_loader
from utils.config import EPSILONS
from utils.reproducibility import set_seed
from utils.evaluation import evaluate
from utils.logging import print_header
from utils.formatting import format_table_row
from utils.metrics import compute_attack_metrics

batch_size = 64

def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial defense on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm', choices=ATTACKS.keys(),
                        help='Attack to evaluate against')
    parser.add_argument('--defense', type=str, default='fgsm', choices=MODELS.keys(),
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

    columns = ['Epsilon', 'Baseline', 'Defended', 'Delta', 'Attack Accuracy']
    widths = [12, 14, 14, 10, 12]
    print_header(columns, widths)

    for epsilon in EPSILONS:
        base_acc = evaluate(base_model, device, test_loader, attack_fn, epsilon)
        def_acc = evaluate(defense_model, device, test_loader, attack_fn, epsilon)
        
        _, _, _, delta, attack_success = compute_attack_metrics(epsilon, base_acc, def_acc).values()
        print(format_table_row([f"{epsilon:.2f}", f"{base_acc:.2f}%", f"{def_acc:.2f}%", f"{delta:.2f}%", f"{attack_success:.2f}%"], widths))

if __name__ == '__main__':
    main()