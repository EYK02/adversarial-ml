# src/attacks/registry.py

from functools import partial
from .fgsm import fgsm_attack
from .pgd import pgd_attack

def get_attack_fn(name, **kwargs):
    if name == "fgsm":
        return fgsm_attack, {}

    if name == "pgd":
        steps = kwargs.get("steps", 10)

        return partial(pgd_attack, steps=steps), {
            "steps": steps
        }
    
    # Add future attacks here

    raise ValueError(f"Unknown attack: {name}")