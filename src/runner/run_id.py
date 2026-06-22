# src/runner/run_id.py

"""
Run identifier generation utilities.

Provides deterministic construction of human-readable experiment IDs
based on configuration metadata (model, dataset, attack, seed, etc.).

These IDs are used for:
- logging directories
- checkpoint paths
- experiment tracking
"""

def make_run_id(
    task: str, 
    model: str, 
    dataset: str, 
    **metadata
) -> str:
    """
    Generate a unique experiment identifier.

    The ID encodes:
    - task type (train, eval, adv_train, etc.)
    - dataset
    - model
    - optional metadata (attack, epsilon, seed, etc.)

    Returns
    -------
    str
        Human-readable but structured run identifier.
    """
    parts = [task, dataset, model]

    key_order = ["defense", "defense_steps", "attack", "steps", "alpha", "epsilon", "seed"]

    ordered = {k: metadata[k] for k in key_order if k in metadata}
    extras  = {k: v for k, v in metadata.items() if k not in key_order}

    for key, value in {**ordered, **extras}.items():
        if value is None:
            continue

        if key == "epsilon":
            parts.append(f"eps{value:.2f}")

        elif key == "seed":
            parts.append(f"seed{value}")

        elif key == "steps":
            parts.append(f"steps{value}")

        elif key == "defense_steps":
            parts.append(f"dsteps{value}")

        elif key == "alpha":
            parts.append(f"alpha{value:.4f}")

        elif isinstance(value, bool):
            if value:
                parts.append(key)

        else:
            parts.append(f"{key}{value}")

    return "_".join(map(str, parts))