# defenses/adversarial_training.py

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from attacks.registry import get_attack_fn
from utils.data import get_mnist_train_loader, get_mnist_test_loader
from utils.logger import JSONLLogger
from utils.reproducibility import set_seed
from utils.evaluation import evaluate
from utils.run_id import make_run_id

start_time = time.perf_counter()

batch_size = 64
epochs = 5
learning_rate = 0.001

training_logger = JSONLLogger("results/jsonl/adv_training.jsonl")
model_logger = JSONLLogger("results/jsonl/model_save.jsonl")

def train_adversarial(model, device, train_loader, optimizer, criterion, attack_fn, epsilon):
    model.train()

    total_loss = 0
    correct_clean = 0
    correct_adv = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        adv_data = attack_fn(model, device, data, target, epsilon)

        combined_data = torch.cat((data, adv_data), dim=0)
        combined_target = torch.cat((target, target), dim=0)

        optimizer.zero_grad()
        output = model(combined_data)
        loss = criterion(output, combined_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted_clean = torch.max(output[:data.size(0)].data, 1)
        _, predicted_adv = torch.max(output[data.size(0):].data, 1)

        correct_clean += (predicted_clean == target).sum().item()
        correct_adv += (predicted_adv == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    clean_acc = 100 * correct_clean / len(train_loader.dataset)
    adv_acc = 100 * correct_adv / len(train_loader.dataset)
    return avg_loss, clean_acc, adv_acc

def main():
    parser = argparse.ArgumentParser(description='Adversarial training on MNIST')
    parser.add_argument('--attack', type=str, default='fgsm',
                        help='Attack to train against')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Perturbation budget for adversarial training')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)
    base_model_path = f'models/cnn_mnist_seed{args.seed}.pth'
    run_id = make_run_id(
        task="adv_train",
        model="cnn_mnist",
        attack=args.attack,
        epsilon=args.epsilon,
        seed=args.seed
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = get_mnist_train_loader(batch_size, seed=args.seed)
    test_loader = get_mnist_test_loader(batch_size)

    defense_model = CNN().to(device)
    defense_model.load_state_dict(torch.load(base_model_path, map_location=device))

    optimizer = optim.Adam(defense_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    attack_fn, attack_params = get_attack_fn(args.attack, steps=args.steps)
    save_path = f'models/adv_{args.attack}_eps_{args.epsilon}_seed{args.seed}.pth'
    print(f"Adversarial training — attack={args.attack}, epsilon={args.epsilon}\n")

    for epoch in range(1, epochs + 1):
        train_loss, clean_acc, adv_acc = train_adversarial(
            defense_model, device, train_loader, optimizer, criterion, attack_fn, args.epsilon
        )
        test_loss, test_acc = evaluate(defense_model, device, test_loader, criterion=criterion)
        
        training_logger.log({
            "run_id":       run_id,
            "run_type":     "adv_training",

            "seed":         args.seed,
            
            "attack":       args.attack,
            "epsilon":      args.epsilon,
            "epoch":        int(epoch),
            
            "train_loss":           float(train_loss),
            "train_clean_accuracy": float(clean_acc),
            "train_adv_accuracy":   float(adv_acc),
            "test_clean_accuracy":  float(test_acc)
        })

    torch.save(defense_model.state_dict(), save_path)
    print(f'\nModel saved to {save_path}')

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    duration = time.perf_counter() - start_time

    model_logger.log({
        "run_id":   run_id,
        "run_type": "model_save",

        "seed": args.seed,

        "attack": args.attack,
        "attack_params": attack_params,
        "epsilon": args.epsilon,

        "duration_sec": duration,

        "model": "adv_cnn_mnist",
        "model_path": save_path
    })

if __name__ == '__main__':
    main()