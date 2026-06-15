# utils/run_id.py

def make_run_id(task, model, attack=None, epsilon=None, seed=None):
    parts = [task, model]

    if attack is not None:
        parts.append(attack)

    if epsilon is not None:
        parts.append(f"eps{epsilon}")

    if seed is not None:
        parts.append(f"seed{seed}")

    return "_".join(map(str, parts))