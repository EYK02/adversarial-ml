# src/utils/config.py

EPSILONS = [0.1, 0.2, 0.3] # 0.0-0.3 step 0.05
BATCH_SIZE = 64

LEARNING_RATE = 0.001
NUM_EPOCHS = 5
NUM_SEEDS = 1 # 5

DEFENSES = [("fgsm", None), ("pgd", 10)] 

# DEFENSES = [
#     ("fgsm", None),
#     ("pgd",  10),
#     ("pgd",  20),
#     ("pgd",  40),
# ]

EVAL_ATTACKS = [
    ("fgsm", None),
    ("pgd", 10) #change back to 40
]