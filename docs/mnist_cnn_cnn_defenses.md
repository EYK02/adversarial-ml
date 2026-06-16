# MNIST - CNN Adversarial Training Defenses

## Setup

- **Model:** CNN baseline (seeds 0-4)
- **Dataset:** MNIST
- **Defenses:** FGSM, PGD (steps 5, 10, 20, 40)
- **Training epsilon:** 0.20 (all defenses)
- **Eval attacks:** FGSM, PGD-40
- **Epsilon range:** 0.00 - 0.30 (step 0.05)
- **Seeds:** 0-4

## Defense robustness vs FGSM

### Baseline vs defense comparison

### Observations

---

## Defense robustness vs PGD-40

### Baseline vs defense comparison

### Observaitons

---

## Cross-evaluation

### Heatmap - epsilon=0.10

### Heatmap - epsilon=0.20

### Heatmap - epsilon=0.30

### Summary table

### Observations

---

## Key findings
---

## Open questions
---

## Caveats
- All defenses trained at epsilon=0.20
- Results are specific to this CNN architecture and MNIST dataset.