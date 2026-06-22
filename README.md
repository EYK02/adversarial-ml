# Adversarial Robustness Framework

A modular, configuration-driven framework for training and evaluating neural networks under adversarial perturbations.

The system is designed for reproducible experimentation across datasets, architectures, training strategies, and adversarial attacks, with support for systematic robustness analysis.

---

## Purpose

This framework provides a controlled environment for studying adversarial robustness in neural networks, including:

* standard and adversarial training
* white-box adversarial attacks (e.g., FGSM, PGD)
* robustness evaluation under varying perturbation strengths
* cross-evaluation across training and attack methods
* reproducible multi-seed experimentation

It is intended as a **general research tool for robustness studies**, not tied to a single dataset or model.

---

## Key Features

* YAML-based experiment configuration
* Modular dataset, model, attack, and training definitions
* Standard and adversarial training pipelines
* FGSM / PGD attack support
* Cross-evaluation between training and evaluation attacks
* Multi-seed reproducibility
* Automatic checkpointing and resume support
* Structured logging and metric tracking
* Analysis and figure generation pipeline

---

## System Overview

Experiments are executed as staged pipelines derived from a single configuration file:

```text
config → experiment expansion → training → evaluation → analysis
```

Each stage is automatically generated and executed by the framework.

---

## Project Structure

```text
adversarial-ml/
├── configs/
│   ├── datasets/        dataset definitions
│   ├── models/          model architectures
│   ├── training/        training configurations
│   ├── attacks/         adversarial attack definitions
│   └── experiments/     full experiment setups
│   
├── src/
│   ├── training/        training loops (standard + adversarial)
│   ├── evaluation/      robustness evaluation
│   ├── attacks/         attack implementations
│   ├── models/          model factory
│   ├── analysis/        reporting and plots
│   └── utils/           config system, logging, seeding, runner
│   
├── scripts/
│   └──run_experiment.py   main CLI entrypoint
│   
├── runs/
│   └── <experiment_name>/
│       ├── logs/
│       ├── metrics/
│       └── figures/
│   
└── checkpoints/            shared model checkpoints
```

---

## Running Experiments

### Environment Setup
#### 1. Clone the repo
```bash
git clone https://github.com/EYK02/adversarial-ml.git
cd adversarial-ml
```
#### 2. Create and activate venv
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```
#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. (Optional) AMD GPU support via ROCm — Linux only
Check your ROCm version, then replace the CPU-only PyTorch from requirements.txt with the matching ROCm wheel:

```bash
bashrocm-smi --version   # e.g. "6.2.0"

pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

Replace rocm6.2 with your installed version. Available builds are listed at pytorch.org/get-started/locally.
Verify the GPU is visible before proceeding:

```bash
bashpython -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output example:
True
AMD Radeon RX 7800 XT

#### 5. Smoke test to verify pipeline
```bash
python -m scripts.run_experiment \
    --config configs/experiments/mnist_cross_eval.yaml \
    --smoke-test
```

#### 6. Dry run to verify full config
```bash
python -m scripts.run_experiment \
    --config configs/experiments/mnist_cross_eval.yaml \
    --dry-run
```

#### 7. Full run
```bash
python -m scripts.run_experiment \
    --config configs/experiments/mnist_cross_eval.yaml
```


## Reproducibility

All experiments are reproducible given:

* fixed configuration file
* fixed random seeds
* deterministic attack definitions

The framework supports multi-seed evaluation to quantify variance across training runs.

---

## Checkpointing & Resume

Adversarial training supports automatic checkpointing:

* `latest.pth` → resume training
* `epoch_XXX.pth` → per-epoch checkpoints
* `best.pth` → best validation model

Interrupted training can be resumed automatically from the latest checkpoint.

---

## Outputs

Each experiment produces structured outputs:

```text
runs/<experiment_name>/
  logs/        training and evaluation logs (JSONL)
  metrics/     aggregated results
  figures/     plots and visualisations
```

Shared checkpoints:

```text
checkpoints/<dataset>/<model>/
```

---

## Example Configuration

```yaml
dataset: configs/datasets/example.yaml
model: configs/models/example.yaml

training:
  - configs/training/standard.yaml
  - configs/training/adv_fgsm.yaml
  - configs/training/adv_pgd10.yaml

eval_attacks:
  - configs/attacks/fgsm.yaml
  - configs/attacks/pgd10.yaml

seeds: [0, 1, 2, 3, 4]

epsilon_eval: [0.0, 0.1, 0.2, 0.3]
```

---

## Analysis

To generate aggregated results and figures:

```bash
python -m src.analysis.report --config configs/experiments/example.yaml
```

---

## Extensibility

The framework is designed to be extended in multiple directions:

* new datasets (e.g. CIFAR-10, ImageNet)
* new architectures (ResNet, WideResNet, etc.)
* new attack methods (black-box, transfer-based)
* new defense strategies
* alternative evaluation protocols

---

## Current Scope

* white-box adversarial attacks
* research-scale experimental workflows
* primarily image classification tasks
* MNIST-based baseline experiments (initial validation)

---

## References

* Goodfellow et al. — *Explaining and Harnessing Adversarial Examples*
* Madry et al. — *Towards Deep Learning Models Resistant to Adversarial Attacks*
* Standard benchmark datasets (e.g., MNIST)

---

## Summary

A general-purpose framework for reproducible adversarial robustness research, supporting systematic comparison of training strategies and evaluation under adversarial attacks.

---
