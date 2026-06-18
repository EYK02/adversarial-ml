# src/logging/run_id.py

def make_run_id(task: str, model: str, **metadata) -> str:
    parts = [task, model]
    
    for key, value in metadata.items():
        if value is None:
            continue

        if key == "epsilon":
            parts.append(f"eps{value}")

        elif key == "seed":
            parts.append(f"seed{value}")

        elif isinstance(value, bool):
            if value:
                parts.append(key)

        elif key == "steps":
            if value is not None:
                parts.append(f"steps{value}")
                
        else:
            parts.append(f"{key}{value}")

    return "_".join(map(str, parts))
