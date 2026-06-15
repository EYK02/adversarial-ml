import argparse
import torch
from model import CNN
from attacks.registry import ATTACKS, MODELS
from utils.data import get_mnist_test_loader
from utils.config import EPSILONS
from utils.reproducibility import set_seed
from utils.evaluation import evaluate_adversarial, evaluate_clean

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

    print(f"Attack: {args.attack} | Defense: {args.defense}\n")
    print(f"|{'Epsilon':<12}|{'Baseline':<14}|{'Defended':<14}|{'Delta':<10}|{'Attack Accuracy':<12}|")
    print("|------------|--------------|--------------|------------|----------------|")

    clean_acc = evaluate_clean(base_model, device, test_loader) # not used?

    for epsilon in EPSILONS:
        base_acc = evaluate_adversarial(base_model, device, test_loader, attack_fn, epsilon)
        def_acc = evaluate_adversarial(defense_model, device, test_loader, attack_fn, epsilon)
        delta = def_acc - base_acc
        sign = "+" if delta >= 0 else ""
        base_str = f"{base_acc:.2f}%"
        def_str = f"{def_acc:.2f}%"
        delta_str = f"{sign}{abs(delta):.2f}%"
        atk_acc_str = f"{100 - base_acc:.2f}%"
        print(f"| {epsilon:<10.2f} | {base_str:<12} | {def_str:<12} | {delta_str:<10}| {atk_acc_str:<12}|")

if __name__ == '__main__':
    main()