# src/attacks/pgd.py

import torch
import torch.nn as nn

def pgd_attack(model, device, images, labels, epsilon, alpha = 0.01, steps=40):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # random start
    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, -1, 1).detach()

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv_images = adv_images.detach().requires_grad_(True)

        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        if adv_images.grad is not None:
            adv_images.grad.zero_()

        loss.backward()

        # gradient step
        adv_images = adv_images + alpha * adv_images.grad.sign()

        # projection back to epsilon ball
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, -1, 1).detach()

    return adv_images