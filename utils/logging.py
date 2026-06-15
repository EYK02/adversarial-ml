# utils/logging.py

from .formatting import format_table_row

def print_header(columns, widths):
    print(format_table_row(columns, widths))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")

def log_epoch_adv(epoch, loss, clean_acc, adv_acc, test_acc=None):
    msg = (
        f"Epoch {epoch} | "
        f"Loss: {loss:.4f} | "
        f"Train Clean: {clean_acc:.2f}% | "
        f"Train Adv: {adv_acc:.2f}%"
    )

    if test_acc is not None:
        msg += f" | Test Clean: {test_acc:.2f}%"

    log(msg)

def log(msg, file=None):
    print(msg)
    if file is not None:
        with open(file, "a") as f:
            f.write(msg + "\n")