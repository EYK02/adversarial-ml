# src/attacks/registry.py

from functools import partial
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.utils.config import AttackConfig


def build_attack(cfg: AttackConfig):

    if cfg.name == "fgsm":
        return fgsm_attack, {}

    if cfg.name == "pgd":

        if cfg.steps is None:
            raise ValueError("PGD requires steps")

        alpha = cfg.alpha

        if alpha is None or alpha == "budget_scaled":
            alpha = 2.5 * cfg.epsilon / cfg.steps

        fn = partial(
            pgd_attack,
            steps=cfg.steps,
            alpha=alpha,
            restarts=cfg.restarts,
        )

        return fn, {
            "steps": cfg.steps,
            "alpha": alpha,
            "restarts": cfg.restarts,
        }
    
    # if cfg,name == "CW":

    raise ValueError(f"Unknown attack: {cfg.name}")


def attack_tag(attack_cfg: AttackConfig) -> str:
    if attack_cfg.steps is not None:
        return f"{attack_cfg.name}{attack_cfg.steps}"
    return attack_cfg.name
