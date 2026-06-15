# Results Log

## FGSM Attack on baseline CNN

**Date:** 2026-06-14
**Model:** CNN trained on MNIST, 98.68% clean accuracy
**Attack:** Fast Gradient Sign Method (FGSM)

|Epsilon|      Test Accuracy|
|------------|--------------|
|0.00|         98.68%|
|0.05|         97.34%|
|0.10|         94.54%|
|0.15|         89.43%|
|0.20|         82.18%|
|0.25|         73.12%|
|0.30|         63.30%|


```bash
python -m attacks.evaluate_attacks --attack fgsm
```

**Observations:**
- Accuracy degrades gradually, but not linearly, and accelerates above epsilon=0.15
- Drop from epsilon=0.0 -> 0.10 is 4.41pp, from epsilon=0.20 -> 0.30 is 18.88pp.

**Open questions:**
- At what epsilon do perturbations become visible to the human eye?
- At what percentage is the model considered unreliable?
- Does adversarial training improve accuracy uniformly across all epsilon values?
- Are certain digits (0-9) classes more vulnerable than others to perturbations?

## FGSM Visualization

**Observations:**
Perturbations appear as grey noise on the black background but cause minimal distortion of the digit itself. The digit remains visually recognizable to the human eye even at epsilon=0.30, yet accuracy of the model drops to 63%.

**Implications:**
A human would still classify correctly where the model fails as to the human eye, the FGSM perturbations are only visible on a uniform background. This confirms that the adversarial vulnerability is model-specific, and not a perceptual ambiguity.

**Further questions:**
- Would an attack concentrated on distorting the digit itself be more or less effective? Against humans, maybe, but maybe not against a model?
- When the model misclassifies, are they random across the digits 0-9, or does it consistently confuse certain digits with specific others? (perhaps 1 and 7?)
- Will the model's confidence decrease as epsilon increases, or does it stay high even when wrong?

## Adversarial Training Defense

**Defense epsilon:** 0.20
**Reason:**
Accuracy drops below 90% at this range and perturbations become more noticeable.

**Questions:**
Does defense training against epsilon=0.20 generalize to other values of epsilon, especially at higher values that the model hasn't seen during training?

## Defense Evaluation - FGSM adversarial Training at epsilon = 0.20

*adversarial_training.py results*
Epoch 1: Loss: 0.0592 | Train Clean: 99.52% | Train Adv: 96.44% | Test Clean: 99.13%
Epoch 2: Loss: 0.0324 | Train Clean: 99.73% | Train Adv: 98.13% | Test Clean: 99.24%
Epoch 3: Loss: 0.0246 | Train Clean: 99.79% | Train Adv: 98.53% | Test Clean: 98.94%
Epoch 4: Loss: 0.0207 | Train Clean: 99.82% | Train Adv: 98.75% | Test Clean: 99.20%
Epoch 5: Loss: 0.0145 | Train Clean: 99.89% | Train Adv: 99.08% | Test Clean: 99.20%

*evaluate_defense.py results*
|Epsilon      |Baseline       |Defended       |Delta     |
| - | - | - | - |
|0.0|          98.68%      |   99.20%|         +0.52      |
|0.05|         97.34%     |    98.91% |        +1.57      |
|0.1  |        94.54%    |     98.68%  |       +4.14     | 
|0.15  |       89.43%   |      98.25%   |      +8.82    |  
|0.2    |      82.18%  |       97.93%    |     +15.75  |   
|0.25    |     73.12% |        97.57%     |    +24.45 |    
|0.3      |    63.30%|         97.17%      |   +33.87|

**Observations:**
- Defense eliminates the majority of FGSM vulnerability across all epsilon values
- Clean accuracy (epsilon=0.0) improved slightly
- Generalization beyond training epsilon (0.20) seems strong as it is effective for at least epsilon=0.25 and 0.3
- Delta grows with epsilon - defense is most valuable where base is weakest

**Possible Caveats:**
- Results may be over-tuned to FGSM and not be effective against other forms of adversarial attacks.
- MNIST is a simple dataset and may not be applicable to harder tasks.
- Near-perfect robustness from 5 epochs is unexpected and warrant further investigation.

**Questions answered:**
- Does adversarial training improve accuracy uniformly? Yes
- Does defense generalise beyond training epsilon? Yes

**Further questions:**
- Are results FGSM-sepcific or general and robust? Test with other attacks (PGD, etc.)

## PGD Attack Evaluation - alpha=0.01, steps=40
|Epsilon     |Baseline      |Defended      |Delta     |
|-|-|-|-|
| 0.00       | 98.68       % | 99.20       % | +0.52    |
| 0.05       | 95.99       % | 98.60       % | +2.61    |
| 0.10       | 86.80       % | 97.41       % | +10.61   |
| 0.15       | 68.48       % | 95.82       % | +27.34   |
| 0.20       | 46.31       % | 92.76       % | +46.45   |
| 0.25       | 28.82       % | 88.99       % | +60.17   |
| 0.30       | 16.30       % | 84.42       % | +68.12   |

**Observations:**
- PGD stronger than FGSM. 16% vs 63% at epsilon=0.30
- Defense against FGSM generalises meaningfully to PGD despite never seeing PGD during training.

**Caveats:**
- Defense not as effective under PGD than FGSM, thus not fully robust
- Adversarial training against PGS would most likely improve PGD robustness further

**Questions answered:**
- Are FGSM defense attack-specific? No, it generalises to PGD
- Is PGD stronger than FGSM? Yes

**Further questions:**
- Would adversarial training against PGD outperform FGSM adversarial training?
- What is the optimal PGD steps, is 40 steps sufficient?
- How does it vary with different alpha values?

### PGD Attack Evaluation - steps=5, 10, 20
```bash
python -m attacks.evaluate_attack --attack pgd5
python -m attacks.evaluate_attack --attack pgd10
python -m attacks.evaluate_attack --attack pgd20
```

**Results:**
*steps=5*
|Epsilon     |Accuracy    |
|-|-|
|0.00        |98.68%|
|0.05        |97.05%|
|0.10        |96.42%|
|0.15        |95.99%|
|0.20        |95.56%|
|0.25        |94.95%|
|0.30        |94.65%|

*steps=10*
|Epsilon     |Accuracy    |
|-|-|
|0.00        |98.68%|
|0.05        |96.17%|
|0.10        |92.80%|
|0.15        |90.72%|
|0.20        |89.03%|
|0.25        |87.30%|
|0.30        |85.51%|

*steps=20*
|Epsilon     |Accuracy    |
|-|-|
|0.00        |98.68%|
|0.05        |96.05%|
|0.10        |87.92%|
|0.15        |78.29%|
|0.20        |70.44%|
|0.25        |62.94%|
|0.30        |56.80%|

*steps=40*
|Epsilon     |Baseline      |
|-|-|
| 0.00       | 98.68       % |
| 0.05       | 95.99       % |
| 0.10       | 86.80       % |
| 0.15       | 68.48       % |
| 0.20       | 46.31       % |
| 0.25       | 28.82       % |
| 0.30       | 16.30       % |

### PGD Defense Evaluation - Attack:pgd10, Defense:pdg5, 10, 20
```bash
python -m defenses.evaluate_defense --attack pgd10 --defense pgd5
python -m defenses.evaluate_defense --attack pgd10 --defense pgd10
python -m defenses.evaluate_defense --attack pgd10 --defense pgd20
```

*defense pgd5*
|Epsilon     |Baseline      |Defended      |Delta     |
|-|-|-|-|
| 0.00       | 98.68%       | 98.96%       | +0.28%    |
| 0.05       | 96.16%       | 98.15%       | +1.99%    |
| 0.10       | 92.94%       | 97.45%       | +4.51%    |
| 0.15       | 90.61%       | 97.18%       | +6.57%    |
| 0.20       | 88.93%       | 96.99%       | +8.06%    |
| 0.25       | 87.43%       | 96.59%       | +9.16%    |
| 0.30       | 85.77%       | 96.32%       | +10.55%   |

*defense pgd10*
|Epsilon     |Baseline      |Defended      |Delta     |
|-|-|-|-|
| 0.00       | 98.68%       | 99.15%       | +0.47%    |
| 0.05       | 96.18%       | 98.62%       | +2.44%    |
| 0.10       | 92.79%       | 98.25%       | +5.46%    |
| 0.15       | 90.72%       | 98.08%       | +7.36%    |
| 0.20       | 88.87%       | 97.98%       | +9.11%    |
| 0.25       | 87.18%       | 97.73%       | +10.55%   |
| 0.30       | 85.56%       | 97.63%       | +12.07%   |

*defense pgd20*
|Epsilon     |Baseline      |Defended      |Delta     |
|-|-|-|-|
| 0.00       | 98.68%       | 99.13%       | +0.45%    |
| 0.05       | 96.19%       | 98.69%       | +2.50%    |
| 0.10       | 92.95%       | 98.40%       | +5.45%    |
| 0.15       | 90.78%       | 98.36%       | +7.58%    |
| 0.20       | 89.11%       | 98.29%       | +9.18%    |
| 0.25       | 87.29%       | 98.25%       | +10.96%   |
| 0.30       | 85.41%       | 98.08%       | +12.67%   |

### Cross-evaluation - PGD defenses against FGSM and vice versa
```bash
python -m defenses.evaluate_defense --attack fgsm --defense pgd10
python -m defenses.evaluate_defense --attack pgd10 --defense fgsm
```
**Results:**

*attack: fgsm, defense: pdg10*
|Epsilon     |Baseline      |Defended      |Delta     |
|-|-|-|-|
| 0.00       | 98.68%       | 99.15%       | +0.47%    |
| 0.05       | 97.34%       | 98.73%       | +1.39%    |
| 0.10       | 94.54%       | 98.36%       | +3.82%    |
| 0.15       | 89.43%       | 97.72%       | +8.29%    |
| 0.20       | 82.18%       | 96.95%       | +14.77%   |
| 0.25       | 73.12%       | 95.70%       | +22.58%   |
| 0.30       | 63.30%       | 94.21%       | +30.91%   |

*attack:pgd10, defense: fgsm*
|Epsilon     |Baseline      |Defended      |Delta     |
|-|-|-|-|
| 0.00       | 98.68%       | 99.20%       | +0.52%    |
| 0.05       | 96.19%       | 98.61%       | +2.42%    |
| 0.10       | 92.87%       | 98.11%       | +5.24%    |
| 0.15       | 90.82%       | 97.71%       | +6.89%    |
| 0.20       | 88.96%       | 97.52%       | +8.56%    |
| 0.25       | 87.41%       | 97.34%       | +9.93%    |
| 0.30       | 85.35%       | 97.05%       | +11.70%   |
