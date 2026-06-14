# Adversarial ML - MNIST

Exploring adversarial attacks and defenses on a simple image classifier using MNIST.

## Goal
Train a CNN on MNIST, implement adversarial attacks (FGSM, eventually PGD and more), and evaluate defenses including adversarial training.

## Structure
- `attacks/` - attack implementation
- `defenses/` - defense implementation
- `models/` - saved model weights
- `notebooks/` - experimentation and visualisation
- `results/` - plots and metrics

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
