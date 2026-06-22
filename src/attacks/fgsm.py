# src/attacks/fgsm.py

"""
Fast Gradient Sign Method (FGSM).
"""

import torch
import torch.nn as nn

def fgsm_attack(model, device, data, target, epsilon):
    """
    Generate adversarial examples using FGSM.

    Parameters
    ----------
    model:
        Target model.

    device:
        Running device.

    data:
        Clean input batch.

    epsilon:
        Maximum perturbation budget.
    """
    data = data.clone().detach().to(device)
    data.requires_grad_(True)

    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = data.grad

    perturbed_data = data + epsilon * data_grad.sign()

    return perturbed_data.detach()
