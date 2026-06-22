import os
from pathlib import Path

def is_colab() -> bool:
    return "COLAB_GPU" in os.environ or "google.colab" in os.sys.modules


def get_base_path() -> Path:
    if is_colab():
        return Path("/content/drive/MyDrive/adversarial_ml_runs")
    return Path(".")