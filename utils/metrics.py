# utils/metrics.py

def compute_attack_metrics(epsilon, base_acc, def_acc=None):
    if def_acc is None:
        return {
            "epsilon": epsilon,
            "accuracy": base_acc
        }

    delta = def_acc - base_acc
    attack_success = 100.0 - base_acc

    return {
        "epsilon": epsilon,
        "baseline": base_acc,
        "defended": def_acc,
        "delta": delta,
        "attack_success": attack_success
    }