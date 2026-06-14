# Results Log

## FGSM Attack on baseline CNN

**Date:** 2026-06-14
**Model:** CNN trained on MNIST, 98.68% clean accuracy
**Attack:** Fast Gradient Sign Method (FGSM)

Epsilon      Test Accuracy
--------------------------
0.00         98.68%
0.05         97.34%
0.10         94.54%
0.15         89.43%
0.20         82.18%
0.25         73.12%
0.30         63.30%

```
python -m attacks.evaluate_fgsm
```
to recreate results.

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
Does defense training against epsilon=0.20 generalize to other values of epsiolon, especially at higher values that the model hasn't seen during training?

## Defense Evaluation - Adversarial Training at epsilon = 0.20

*adversarial_training.py results*
Epoch 1: Loss: 0.0592 | Train Clean: 99.52% | Train Adv: 96.44% | Test Clean: 99.13%
Epoch 2: Loss: 0.0324 | Train Clean: 99.73% | Train Adv: 98.13% | Test Clean: 99.24%
Epoch 3: Loss: 0.0246 | Train Clean: 99.79% | Train Adv: 98.53% | Test Clean: 98.94%
Epoch 4: Loss: 0.0207 | Train Clean: 99.82% | Train Adv: 98.75% | Test Clean: 99.20%
Epoch 5: Loss: 0.0145 | Train Clean: 99.89% | Train Adv: 99.08% | Test Clean: 99.20%

*evaluate_defense.py results*
Epsilon      Baseline       Defended       Delta     
--------------------------------------------------
0.0          98.68%         99.20%         +0.52      
0.05         97.34%         98.91%         +1.57      
0.1          94.54%         98.68%         +4.14      
0.15         89.43%         98.25%         +8.82      
0.2          82.18%         97.93%         +15.75     
0.25         73.12%         97.57%         +24.45     
0.3          63.30%         97.17%         +33.87

**Observations:**
- Defense eliminates the majority of FGSM vulnerability across all epsilon values
- Clean accuracy (epsilon=0.0) improved slightly
- Generalization beyond training epsilon (0.20) seems strong as it is effective for at least epsilon=0.25 and 0.3
- Delta grows with epsilon - defense is most valuable where base is weakest

**Possible Caveats:**
- Results may be over-tuned to FGSM and not be effective against other forms of adversarial attacks.
- MNIST is a simple dataset and may not be applicable to harder tasks.

**Questions answered:**
- Does adversarial training improve accuracy uniformly? Yes
- Does defense generalise beyond trainig epsilon? Yes

**Further questions:**
- Are results FGSM-sepcific or general and rubust? Test with other attacks (PGD, etc.)