import argparse
import torch
from model import CNN
from attacks.registry import ATTACKS
from utils.data import get_mnist_test_loader
from utils.config import EPSILONS
from utils.reproducibility import set_seed
from utils.evaluation import evaluate
batch_size = 64

def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial attack on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm', choices=ATTACKS.keys(), help='Attack to evaluate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model_path = f'models/cnn_mnist_seed{args.seed}.pth'

    model = CNN().to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    model.eval()

    attack_fn = ATTACKS[args.attack]
    test_loader = get_mnist_test_loader(batch_size)

    print(f"Evaluating {args.attack.upper()} attack on base model\n")
    print(f"|{'Epsilon':<12}|{'Accuracy':<12}|")
    print("|------------|------------|")

    for epsilon in EPSILONS:
        acc = evaluate(model, device, test_loader, attack_fn, epsilon)
        print(f"|{epsilon:<12.2f}|{acc:<.2f}%|")

if __name__ == '__main__':
    main()