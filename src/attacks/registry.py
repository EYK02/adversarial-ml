# attacks/registry.py

from functools import partial
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

def get_attack_fn(name, **kwargs):
    if name == "fgsm":
        return fgsm_attack, {}

    if name == "pgd":
        steps = kwargs.get("steps", 10)

        return partial(pgd_attack, steps=steps), {
            "steps": steps
        }

    raise ValueError(f"Unknown attack: {name}")