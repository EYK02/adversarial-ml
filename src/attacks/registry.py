# src/attacks/registry.py

from functools import partial
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.config import AttackConfig


def build_attack(spec: AttackConfig):

    if spec.name == "fgsm":
        return fgsm_attack, {}

    if spec.name == "pgd":

        if spec.steps is None:
            raise ValueError("PGD requires steps")

        alpha = spec.alpha

        if alpha is None or alpha == "budget_scaled":
            alpha = 2.5 * spec.epsilon / spec.steps

        fn = partial(
            pgd_attack,
            steps=spec.steps,
            alpha=alpha,
            restarts=spec.restarts,
        )

        return fn, {
            "steps": spec.steps,
            "alpha": alpha,
            "restarts": spec.restarts,
        }

    raise ValueError(f"Unknown attack: {spec.name}")


def attack_tag(attack_cfg: AttackConfig) -> str:
    if attack_cfg.steps is not None:
        return f"{attack_cfg.name}{attack_cfg.steps}"
    return attack_cfg.name
