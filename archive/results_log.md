# Results Log

## 1. Baseline CNN

**Date:** 2026-06-14  
**Model:** CNN trained on MNIST  
**Clean accuracy:** 98.68%

---

## 2. FGSM Attack

**Evaluation command:**
```bash
python -m attacks.evaluate_attack --attack fgsm
```

| Epsilon | Accuracy |
|---------|----------|
| 0.00    | 98.68%   |
| 0.05    | 97.34%   |
| 0.10    | 94.54%   |
| 0.15    | 89.43%   |
| 0.20    | 82.18%   |
| 0.25    | 73.12%   |
| 0.30    | 63.30%   |

**Observations:**
- Accuracy degrades gradually but not linearly, accelerating above epsilon=0.15
- Drop from epsilon=0.00 to 0.10 is 4.14pp vs 18.88pp from epsilon=0.20 to 0.30

**Visualisation observations:**
- Perturbations appear as grey noise on the black background with minimal distortion of the digit itself
- Digits remain visually recognisable to a human at epsilon=0.30, yet model accuracy drops to 63%
- Adversarial vulnerability is model-specific, not a perceptual ambiguity — a human would still classify correctly where the model fails

---

## 3. PGD Attack

**Evaluation commands:**
```bash
python -m attacks.evaluate_attack --attack pgd5
python -m attacks.evaluate_attack --attack pgd10
python -m attacks.evaluate_attack --attack pgd20
python -m attacks.evaluate_attack --attack pgd40
```

| Epsilon | pgd5   | pgd10  | pgd20  | pgd40  |
|---------|--------|--------|--------|--------|
| 0.00    | 98.68% | 98.68% | 98.68% | 98.68% |
| 0.05    | 97.05% | 96.17% | 96.05% | 95.99% |
| 0.10    | 96.42% | 92.80% | 87.92% | 86.80% |
| 0.15    | 95.99% | 90.72% | 78.29% | 68.48% |
| 0.20    | 95.56% | 89.03% | 70.44% | 46.31% |
| 0.25    | 94.95% | 87.30% | 62.94% | 28.82% |
| 0.30    | 94.65% | 85.51% | 56.80% | 16.30% |

**Observations:**
- PGD is substantially stronger than FGSM — pgd40 drops baseline to 16% at epsilon=0.30 vs 63% under FGSM
- Attack strength scales with steps but non-linearly — the gap between 20 and 40 steps (56% to 16% at epsilon=0.30) is far larger than between 5 and 10 steps (94% to 85%)
- pgd5 is a weak attack, barely degrading accuracy even at epsilon=0.30 (94.65%)
- 40 steps is a meaningfully stronger evaluation than 20 — attack has not converged at 20 steps

---

## 4. FGSM Adversarial Training Defense

**Training command:**
```bash
python -m defenses.adversarial_training --attack fgsm --epsilon 0.20
```

**Rationale for epsilon=0.20:** Accuracy drops below 90% at this range and perturbations become more noticeable to the human eye.

**Training results:**

| Epoch | Loss   | Train Clean | Train Adv | Test Clean |
|-------|--------|-------------|-----------|------------|
| 1     | 0.0592 | 99.52%      | 96.44%    | 99.13%     |
| 2     | 0.0324 | 99.73%      | 98.13%    | 99.24%     |
| 3     | 0.0246 | 99.79%      | 98.53%    | 98.94%     |
| 4     | 0.0207 | 99.82%      | 98.75%    | 99.20%     |
| 5     | 0.0145 | 99.89%      | 99.08%    | 99.20%     |

**Observations:**
- Gap between clean and adversarial training accuracy closed from 3.08pp (epoch 1) to 0.81pp (epoch 5)
- Clean test accuracy unchanged — no robustness/accuracy tradeoff observed
- Near-perfect robustness from 5 epochs of fine-tuning is unexpected and warrants further investigation

### 4a. FGSM Defense vs FGSM Attack

**Evaluation command:**
```bash
python -m defenses.evaluate_defense --attack fgsm --defense fgsm
```

| Epsilon | Baseline | Defended | Delta    |
|---------|----------|----------|----------|
| 0.00    | 98.68%   | 99.20%   | +0.52%   |
| 0.05    | 97.34%   | 98.91%   | +1.57%   |
| 0.10    | 94.54%   | 98.68%   | +4.14%   |
| 0.15    | 89.43%   | 98.25%   | +8.82%   |
| 0.20    | 82.18%   | 97.93%   | +15.75%  |
| 0.25    | 73.12%   | 97.57%   | +24.45%  |
| 0.30    | 63.30%   | 97.17%   | +33.87%  |

**Observations:**
- Defense nearly eliminates FGSM vulnerability across all epsilon values
- Generalisation beyond training epsilon (epsilon=0.20) is strong — effective at epsilon=0.25 and epsilon=0.30
- Delta grows with epsilon — defense most valuable where baseline is weakest

### 4b. FGSM Defense vs PGD Attack (steps=40)

**Evaluation command:**
```bash
python -m defenses.evaluate_defense --attack pgd40 --defense fgsm
```

| Epsilon | Baseline | Defended | Delta    |
|---------|----------|----------|----------|
| 0.00    | 98.68%   | 99.20%   | +0.52%   |
| 0.05    | 96.01%   | 98.60%   | +2.59%   |
| 0.10    | 86.89%   | 97.41%   | +10.52%  |
| 0.15    | 68.20%   | 95.82%   | +27.62%  |
| 0.20    | 46.03%   | 92.87%   | +46.84%  |
| 0.25    | 28.55%   | 88.98%   | +60.43%  |
| 0.30    | 16.30%   | 84.63%   | +68.33%  |

**Observations:**
- FGSM defense generalises meaningfully to PGD despite never seeing PGD during training
- Defense less effective under PGD than FGSM — drops to 84.63% vs 97.17% at epsilon=0.30
- Still does substantial work pulling model from near-random (16%) to useful (84%)

---

## 5. PGD Adversarial Training Defense

**Training commands:**
```bash
python -m defenses.adversarial_training --attack pgd5 --epsilon 0.15
python -m defenses.adversarial_training --attack pgd10 --epsilon 0.15
python -m defenses.adversarial_training --attack pgd20 --epsilon 0.15
python -m defenses.adversarial_training --attack pgd40 --epsilon 0.15
```

**Rationale for epsilon=0.15:** PGD is a stronger attack — baseline drops below 90% earlier than FGSM, making epsilon=0.15 the equivalent stress point.

### 5a. PGD Defenses vs PGD40 Attack

**Evaluation commands:**
```bash
python -m defenses.evaluate_defense --attack pgd40 --defense pgd5
python -m defenses.evaluate_defense --attack pgd40 --defense pgd10
python -m defenses.evaluate_defense --attack pgd40 --defense pgd20
python -m defenses.evaluate_defense --attack pgd40 --defense pgd40
```

| Epsilon | Baseline | pgd5 def | pgd10 def | pgd20 def | pgd40 def |
|---------|----------|----------|-----------|-----------|-----------|
| 0.00    | 98.68%   | 98.96%   | 99.15%    | 99.13%    | 99.06%    |
| 0.05    | 95.99%   | 98.14%   | 98.59%    | 98.66%    | 98.60%    |
| 0.10    | 86.87%   | 96.66%   | 97.69%    | 98.16%    | 97.99%    |
| 0.15    | 68.34%   | 94.15%   | 96.46%    | 97.30%    | 97.36%    |
| 0.20    | 46.25%   | 89.66%   | 94.14%    | 96.30%    | 96.72%    |
| 0.25    | 28.71%   | 83.73%   | 91.01%    | 94.78%    | 95.60%    |
| 0.30    | 16.29%   | 76.46%   | 86.95%    | 93.38%    | 94.70%    |

**Key findings:** Any adersarial training is clearly better than none. We can see that for the MNIST dataset, running more iterations or steps in the training data does not contribute to any substantial improvement to accuracy, even at higher epsilon. For twice the number of iterations (steps=20 to 40) we see a delta of +1.32% between pgd20 and pgd40 (epsilon=0.30).

**Caveats:** Results may vary if similar training and evaluation is applied to a harder dataset such as CIFAR-10.

---

## 6. Cross-Evaluation

**Evaluation commands:**
```bash
python -m defenses.evaluate_defense --attack fgsm --defense pgd40
python -m defenses.evaluate_defense --attack pgd40 --defense fgsm
```

| Epsilon | fgsm def / pgd40 atk | pgd40 def / fgsm atk |
|---------|----------------------|----------------------|
| 0.00    | 99.20%               | 99.06%               |
| 0.05    | 98.60%               | 98.75%               |
| 0.10    | 97.41%               | 98.44%               |
| 0.15    | 95.82%               | 97.97%               |
| 0.20    | 92.87%               | 97.58%               |
| 0.25    | 88.98%               | 97.19%               |
| 0.30    | 84.63%               | 96.62%               |

**Key finding:** PGD40-trained defense generalises better to FGSM (96.62% at epsilon=0.30) than FGSM-trained defense generalises to PGD40 (84.63%). Training against a stronger, more thorough attack produces more general robustness.

---

## 7. Open Questions

- At what percentage is a model considered unreliable?
- Are certain digit classes more vulnerable than others to perturbations?
- When misclassifying, does the model confuse specific digit pairs consistently (e.g. 1 and 7)?
- Does model confidence decrease as epsilon increases, or stay high even when wrong?
- Would an attack concentrated on digit pixels rather than background be more effective?
- Why does PGD defense generalise better to FGSM than FGSM defense generalises to PGD — is training attack strength the determining factor?
- Does the marginal improvement from more PGD training steps justify the training time cost in practice?
- Why does adversarial training in improve clena accuracy (epsilon=0.0), is this a regularisation effect?
- Would these results hold on a harder dataset like CIFAR-10?

---

## 8. Caveats

- MNIST is a simple dataset and results may not generalise to harder tasks or different architectures
- All defenses show near-perfect robustness against FGSM, which is unexpectedly strong and warrants further investigation, particularly whether this holds on harder datasets
- PGD defense generalises better across attack types than FGSM defense, suggesting training attack strength is a meaningful factor in robustness generalisation — but this conclusion is based on one architecture and one dataset
- Small baseline variance (~0.03%) across PGD runs due to random initialisation, but does not affect conclusions
- Results are specific to this CNN architecture and may differ with deeper or more complex models
- PGD attack has not fully converged at 40 steps for all epsilon values — a stronger evaluation might require more steps at higher epsilon
