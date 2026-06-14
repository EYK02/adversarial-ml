from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

ATTACKS = {
    'fgsm': lambda model, device, data, target, eps: fgsm_attack(
        model, device, data, target, eps
    ),
    'pgd5': lambda model, device, data, target, eps: pgd_attack(
        model, device, data, target, eps, alpha=0.01, iters=5
    ),
    'pgd10': lambda model, device, data, target, eps: pgd_attack(
        model, device, data, target, eps, alpha=0.01, iters=10
    ),
    'pgd20': lambda model, device, data, target, eps: pgd_attack(
        model, device, data, target, eps, alpha=0.01, iters=20
    ),
    'pgd40': lambda model, device, data, target, eps: pgd_attack(
        model, device, data, target, eps, alpha=0.01, iters=40
    ),
}