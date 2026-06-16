# defenses/adversarial_training.py

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.factory import load_model
from src.attacks.registry import get_attack_fn
from src.data.loader import get_mnist_train_loader, get_mnist_test_loader
from src.logging.logger import JSONLLogger
from src.utils.reproducibility import set_seed
from src.evaluation.core import evaluate
from src.logging.run_id import make_run_id

batch_size    = 64
epochs        = 5
learning_rate = 0.001

training_logger = JSONLLogger("results/jsonl/adv_training.jsonl")
model_logger    = JSONLLogger("results/jsonl/model_save.jsonl")


def train_adversarial(model, device, train_loader, optimizer, criterion, attack_fn, epsilon):
    model.train()

    total_loss    = 0
    correct_clean = 0
    correct_adv   = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        adv_data = attack_fn(model, device, data, target, epsilon)

        combined_data   = torch.cat((data, adv_data), dim=0)
        combined_target = torch.cat((target, target), dim=0)

        optimizer.zero_grad()
        output = model(combined_data)
        loss   = criterion(output, combined_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted_clean = torch.max(output[:data.size(0)].data, 1)
        _, predicted_adv   = torch.max(output[data.size(0):].data, 1)

        correct_clean += (predicted_clean == target).sum().item()
        correct_adv   += (predicted_adv   == target).sum().item()

    avg_loss  = total_loss / len(train_loader)
    clean_acc = 100 * correct_clean / len(train_loader.dataset)
    adv_acc   = 100 * correct_adv   / len(train_loader.dataset)

    return avg_loss, clean_acc, adv_acc


def main():
    parser = argparse.ArgumentParser(description="Adversarial training on MNIST")
    parser.add_argument("--attack",   type=str,   default="fgsm", help="Attack to train against")
    parser.add_argument("--epsilon",  type=float, default=0.2,    help="Perturbation budget")
    parser.add_argument("--steps",    type=int,   default=None,   help="PGD steps (PGD only)")
    parser.add_argument("--seed",     type=int,   default=0,      help="Random seed")
    args = parser.parse_args()

    start_time = time.perf_counter()

    set_seed(args.seed)

    run_id = make_run_id(
        task="adv_train",
        model="cnn_mnist",
        attack=args.attack,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    # model naming: adv_fgsm_eps0.2_seed0.pth / adv_pgd10_eps0.2_seed0.pth
    if args.attack == "pgd" and args.steps is not None:
        attack_tag = f"pgd{args.steps}"
    else:
        attack_tag = args.attack

    base_model_path = f"models/cnn_mnist_seed{args.seed}.pth"
    save_path = f"models/cnn_mnist_adv_{attack_tag}_eps{args.epsilon}_seed{args.seed}.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_mnist_train_loader(batch_size, seed=args.seed)
    test_loader  = get_mnist_test_loader(batch_size)

    defense_model = load_model(base_model_path, device)
    optimizer = optim.Adam(defense_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # fix: use args.steps, not args.steps on undefined parser arg
    attack_fn, attack_params = get_attack_fn(args.attack, steps=args.steps)

    print(f"Adversarial training — attack={attack_tag}, epsilon={args.epsilon}, seed={args.seed}\n")

    for epoch in range(1, epochs + 1):
        train_loss, clean_acc, adv_acc = train_adversarial(
            defense_model, device, train_loader, optimizer, criterion, attack_fn, args.epsilon
        )
        _, test_acc = evaluate(defense_model, device, test_loader, criterion=criterion)

        print(f"Epoch {epoch}: loss={train_loss:.4f}  clean={clean_acc:.2f}%  adv={adv_acc:.2f}%  test={test_acc:.2f}%")

        training_logger.log({
            "run_id":   run_id,
            "run_type": "adv_training",
            "model":    "cnn_mnist",
            "dataset":  "mnist",
            "seed":     args.seed,

            "attack":        args.attack,
            "attack_params": attack_params,
            "epsilon":       args.epsilon,
            "epoch":         int(epoch),

            "train_loss":           float(train_loss),
            "train_clean_accuracy": float(clean_acc),
            "train_adv_accuracy":   float(adv_acc),
            "test_clean_accuracy":  float(test_acc),
        })

    torch.save(defense_model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    duration = time.perf_counter() - start_time

    model_logger.log({
        "run_id":   run_id,
        "run_type": "model_save",
        "model":    "cnn_mnist",
        "dataset":  "mnist",
        "seed":     args.seed,

        "attack":        args.attack,
        "attack_params": attack_params,
        "epsilon":       args.epsilon,

        "model_path":   save_path,
        "duration_sec": float(duration),
    })


if __name__ == "__main__":
    main()