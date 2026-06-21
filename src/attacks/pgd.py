# src/attacks/pgd.py

import torch
import torch.nn as nn


def pgd_attack(
        model,
        device,
        images,
        labels,
        epsilon,
        alpha   = 0.01,
        steps   = 40,
        restarts = 1,
):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    loss_fn = nn.CrossEntropyLoss()

    worst_adv    = images.clone()
    worst_loss   = torch.full((images.size(0),), float("-inf"), device=device)

    for _ in range(restarts):
        # random start within epsilon ball
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        for _ in range(steps):
            adv_images = adv_images.detach().requires_grad_(True)

            outputs = model(adv_images)
            loss    = loss_fn(outputs, labels)

            model.zero_grad()
            loss.backward()

            # gradient step
            adv_images = adv_images + alpha * adv_images.grad.sign()

            # project back to epsilon ball around original images
            perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images   = torch.clamp(images + perturbation, 0, 1).detach()

        # keep worst-case across restarts (per image)
        with torch.no_grad():
            per_image_loss = nn.CrossEntropyLoss(reduction="none")(
                model(adv_images), labels
            )
            improved          = per_image_loss > worst_loss
            worst_loss        = torch.where(improved, per_image_loss, worst_loss)
            worst_adv[improved] = adv_images[improved]

    return worst_adv