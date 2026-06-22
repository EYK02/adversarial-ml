# src/attacks/registry.py

"""
Attack registry.

Provides a unified interface for constructing adversarial attacks
from AttackConfig objects.

External modules should use build_attack() rather than importing
individual attack implementations directly.
"""

from functools import partial
from typing import Callable
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.config import AttackConfig


def build_attack(
    attack_cfg: AttackConfig
) -> tuple[Callable, dict]:
    """
    Build an attack function from an AttackConfig.

    Parameters
    ----------
    attack_cfg:
        Attack configuration.

    Returns
    -------
    attack_fn:
        Callable with signature:

            attack_fn(
                model,
                device,
                images,
                labels,
                epsilon,
            )

    attack_params:
        Dictionary containing resolved attack parameters.
    """
    if attack_cfg.name == "fgsm":
        return fgsm_attack, {}

    if attack_cfg.name == "pgd":

        if attack_cfg.steps is None:
            raise ValueError("PGD requires steps")

        alpha = attack_cfg.alpha

        if alpha is None or alpha == "budget_scaled":
            alpha = 2.5 * attack_cfg.epsilon / attack_cfg.steps

        fn = partial(
            pgd_attack,
            steps=attack_cfg.steps,
            alpha=alpha,
            restarts=attack_cfg.restarts,
        )

        return fn, {
            "steps": attack_cfg.steps,
            "alpha": alpha,
            "restarts": attack_cfg.restarts,
        }
    
    # if cfg,name == "CW":

    raise ValueError(f"Unknown attack: {attack_cfg.name}")


def attack_tag(
    attack_cfg: AttackConfig,
) -> str:
    """
    Generate a human-readable identifier for an attack.

    Examples
    --------
    fgsm -> "fgsm"
    pgd (steps=10) -> "pgd10"
    """
    if attack_cfg.steps is not None:
        return f"{attack_cfg.name}{attack_cfg.steps}"
    return attack_cfg.name
