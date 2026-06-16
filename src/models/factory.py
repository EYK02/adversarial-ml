# src/models/factory.py

import torch
from .cnn import CNN


def load_model(base_model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    model.eval()

    return model