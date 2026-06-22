# src/attacks/registry.py

from functools import partial
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

def build_attack(spec):
    if spec.name == "fgsm":
        return fgsm_attack, {}

    if spec.name == "pgd":
        if spec.steps is None:
            raise ValueError("PGD requires 'steps' in AttackConfig")
        
        if spec.alpha is None:
            # resolve here, NOT at call site
            spec.alpha = 2.5 * spec.epsilon / spec.steps

        fn = partial(
            pgd_attack,
            steps=spec.steps,
            alpha=spec.alpha,
            restarts=spec.restarts,
        )

        return fn, {
            "steps": spec.steps,
            "alpha": spec.alpha,
            "restarts": spec.restarts,
        }

    raise ValueError(f"Unknown attack: {spec.name}")
