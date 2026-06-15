# utils/formatting.py

def format_table_row(values, widths):
    return "| " + " | ".join(
        f"{str(v):<{w}}" for v, w in zip(values, widths)
    ) + " |"
