# src/training/train.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import CNN
from datasets.loader import get_mnist_train_loader, get_mnist_test_loader
from utils.seed import set_seed
from old.src.utils.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from evaluation.core import evaluate
from utils.logger import JSONLLogger

training_logger = JSONLLogger("artifacts/jsonl/training.jsonl")   
model_logger = JSONLLogger("artifacts/jsonl/model_save.jsonl")    

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy

def main():
    os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    set_seed(args.seed)

    model_save_path = f'models/cnn_mnist_seed{args.seed}.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_mnist_train_loader(BATCH_SIZE, seed=args.seed)
    test_loader = get_mnist_test_loader(BATCH_SIZE)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)    # Optimizer: Adam

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print("[LOAD] Existing model found, skipping training.")
        return

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion=criterion)

        training_logger.log({
            "run_id":       "cnn_mnist_clean",
            "run_type":     "training",

            "dataset":      "mnist",
            "model":        "cnn_mnist",

            "seed":         args.seed,
            "epoch":        int(epoch+1),
            
            "train_loss":       float(train_loss),
            "train_accuracy":   float(train_acc),
            "test_loss":        float(test_loss),
            "test_accuracy":    float(test_acc)
        })

    torch.save(model.state_dict(), model_save_path)

    model_logger.log({
        "run_id":       f"clean_cnn_mnist_seed{args.seed}",
        "run_type":     "model_save",
        "device":       str(device),
        "seed":         args.seed,

        "model":        "cnn_mnist",
        "path":         model_save_path,

        "BATCH_SIZE":       int(BATCH_SIZE),
        "num_epochs":       int(NUM_EPOCHS),
        "optimizer":        "adam",
        "LEARNING_RATE":    LEARNING_RATE
    })
    

if __name__ == '__main__':
    main()