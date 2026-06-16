# attacks/fgsm.py

import torch
import torch.nn as nn

def fgsm_attack(model, device, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, -1, 1)
    return perturbed_data