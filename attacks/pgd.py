import torch
import torch.nn as nn

def pgd_attack(model, device,images, labels, epsilon, alpha=0.01, iters=40):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=-1, max=1).detach()
    
    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        model.zero_grad()
        cost = nn.CrossEntropyLoss()(outputs, labels)
        cost.backward()

        with torch.no_grad():
            adv_images = adv_images + alpha*adv_images.grad.sign()
            perturbation = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
            adv_images = torch.clamp(images + perturbation, min=-1, max=1).detach()
    
    return adv_images
