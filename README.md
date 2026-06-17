# Adversarial ML

Does the way you train defense determine how well it generalises to attacks it has never seem?

Exploring adversarial attacks and defenses on CNN image classifiers.

## Overview

A from-scratch adversarial ML sandbox in Python. The project covers the full
pipeline: CNN classifier training across multiple seeds, adversarial attacks
(FGSM and PGD), adversarial training defenses, cross-evaluation between
attack/defense types, and statistical analysis across seeds. Everything is
CLI-driven with a clean src/ layout and a structured logging and analysis
pipeline.

## Repo structure
```
adversarial-ml/
├── run_all.py                  # reproducible full pipeline across all seeds
├── runs/                       # individual stage sweep scripts
│   ├── run_training.py
│   ├── run_attack_eval.py
│   ├── run_adversarial_training.py
│   └── run_defense_eval.py
├── src/
│   ├── attacks/                # FGSM, PGD implementations
│   ├── training/               # baseline and adversarial training
│   ├── evaluation/             # attack eval, defense eval
│   ├── models/                 # CNN architecture, model loading
│   ├── data/                   # MNIST loader
│   ├── logging/                # JSONL logger, run ID
│   └── utils/                  # config, reproducibility, runner
├── analysis/                   # log loading, normalization, aggregation, plots
│   ├── schema.py               # authoritative normalized schema
│   ├── load_logs.py
│   ├── normalize.py
│   ├── aggregate.py
│   ├── plots.py
│   └── report.py               # produces artifacts/
├── artifacts/
│   ├── jsonl/                  # raw experiment logs
│   ├── images/                 # plots
│   └── csv/                    # summary tables
├── tools/                      #
│   ├── visualize_attacks.py    # produces visualization of attack effects
├── models/                     # model storage location
└── docs/
    ├── mnist_cnn_baseline.md
    ├── mnist_cnn_attacks.md
    └── mnist_cnn_defenses.md
```
## Setup

```bash
git clone https://github.com/EYK02/adversarial-ml.git
cd adversarial-ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing results

Run the full pipeline, train → attack → defend → analyze:
```bash
python run_all.py
```

Or run individual stages:
```bash
python -m runs.run_training
python -m runs.run_attack_eval
python -m runs.run_adversarial_training
python -m runs.run_defense_eval
```

Generate analysis report:
```bash
python -m analysis.report
```

## Stack

Python 3.12, PyTorch (CPU), MNIST — Windows

## Findings

See docs/ for detailed findings per experiment.

## Open questions

- Are certain digit classes more vulnerable than others?
- Does model confidence stay high when wrong under attack?
- Why does PGD defense generalise better to FGSM than FGSM defense to PGD?
- Would curriculum adversarial training (FGSM → PGD) outperform single-attack training?
- Would results hold on CIFAR-10?
- Is the marginal improvement from more PGD steps worth the training cost?
- Why does adversarial training slightly improve clean accuracy — regularisation effect?
- Would an attack concentrated on digit pizels rather than background be more effective?