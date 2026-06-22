# src/models/factory.py

import torch
from src.models.cnn import CNN
from src.utils.config import ModelConfig


# ─────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────

_MODELS = {
    "cnn": CNN,
}


def _get_model_class(name: str):
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(_MODELS)}")
    return _MODELS[name]


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────

def create_model(cfg: ModelConfig, device: torch.device) -> torch.nn.Module:
    """Instantiate a fresh model from config."""
    cls = _get_model_class(cfg.name)
    return cls().to(device)


def load_model(path: str, device: torch.device, cfg: ModelConfig) -> torch.nn.Module:
    """Load a model from a checkpoint."""
    model = create_model(cfg, device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_or_create_model(cfg: ModelConfig, device: torch.device) -> torch.nn.Module:
    """Always returns a fresh untrained model — used when no checkpoint exists yet."""
    return create_model(cfg, device)
