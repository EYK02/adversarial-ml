# src/attacks/registry.py

from functools import partial
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


def get_attack_fn(name: str, **kwargs):
    """
    Returns (attack_fn, attack_params) where:
    - attack_fn  : callable (model, device, images, labels, epsilon) -> adv_images
    - attack_params : dict of resolved parameters for logging
    """
    if name == "fgsm":
        return fgsm_attack, {}

    if name == "pgd":
        steps    = kwargs.get("steps",    10)
        alpha    = kwargs.get("alpha",    None)
        restarts = kwargs.get("restarts", 1)

        if alpha is None:
            raise ValueError(
                "pgd alpha must be resolved before calling get_attack_fn. "
                "Use alpha = 2.5 * epsilon / steps at the call site."
            )

        fn = partial(pgd_attack, steps=steps, alpha=alpha, restarts=restarts)

        return fn, {
            "steps":    steps,
            "alpha":    alpha,
            "restarts": restarts,
        }

    raise ValueError(f"Unknown attack: {name!r}. Available: fgsm, pgd")