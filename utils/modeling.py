# utils/modeling.py

import torch
from model import CNN

def load_model(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model