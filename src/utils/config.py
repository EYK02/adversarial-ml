# src/utils/config.py

BATCH_SIZE = 64
LEARNING_RATE = 0.001

NUM_EPOCHS = 5
NUM_SEEDS = 5 

EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] 

PGD_STEPS = [10, 20, 40]

DEFENSES = [("fgsm", None)] + [("pgd", step) for step in PGD_STEPS]
ATTACKS = [("fgsm", None)] + [("pgd", step) for step in PGD_STEPS]

EVAL_ATTACKS = [("fgsm", None), ("pgd", 40)]

# Dry run config
NUM_SEEDS_DRY = 1
EPSILONS_DRY =  [0.1, 0.2, 0.3]
DEFENSES_DRY = [("fgsm", None), ("pgd", 10)]
EVAL_ATTACKS_DRY = [("fgsm", None), ("pgd", 10)]
