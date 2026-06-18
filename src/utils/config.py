# src/utils/config.py

#EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] 
EPSILONS =  [0.1, 0.2, 0.3] # dry
BATCH_SIZE = 64

LEARNING_RATE = 0.001
NUM_EPOCHS = 5
#NUM_SEEDS = 5 
NUM_SEEDS = 1 # dry


# DEFENSES = [
#     ("fgsm", None),
#     ("pgd",  10),
#     ("pgd",  20),
#     ("pgd",  40),
# ]

DEFENSES = [("fgsm", None), ("pgd", 10)]  #dry run

#EVAL_ATTACKS = [("fgsm", None), ("pgd", 40)]

EVAL_ATTACKS = [("fgsm", None), ("pgd", 10)] #dry