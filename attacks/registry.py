# attacks/registry.py

from functools import partial
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

ATTACKS = {
    'fgsm': fgsm_attack,
    'pgd':  pgd_attack   
}

def get_attack_fn(name, args):
    if name == "fgsm":
        return fgsm_attack, {}

    if name == "pgd":
        return partial(pgd_attack, steps=args.steps), {
            "steps": args.steps
        }