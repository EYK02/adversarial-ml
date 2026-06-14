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