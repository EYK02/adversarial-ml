# utils/logging.py

from typing import Optional

def format_table_row(values, widths):
    return "| " + " | ".join(
        f"{str(v):<{w}}" for v, w in zip(values, widths)
    ) + " |"


def print_header(columns, widths):
    print(format_table_row(columns, widths))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")


def log_attack_results(epsilon, base_acc, def_acc=None):
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

def log_epoch_adv(epoch, loss, clean_acc, adv_acc, test_acc=None):
    msg = (
        f"Epoch {epoch} | "
        f"Loss: {loss:.4f} | "
        f"Train Clean: {clean_acc:.2f}% | "
        f"Train Adv: {adv_acc:.2f}%"
    )

    if test_acc is not None:
        msg += f" | Test Clean: {test_acc:.2f}%"

    print(msg)