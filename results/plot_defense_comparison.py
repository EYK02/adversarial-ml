import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
save_path = 'results/images/defense_comparison.png'

# Results from evaluate_defense --attack pgd40
RESULTS = {
    'base':  [98.68, 95.99, 86.80, 68.48, 46.31, 28.82, 16.30],
    'fgsm':  [99.20, 98.60, 97.41, 95.82, 92.87, 88.98, 84.63],
    'pgd5':  [98.96, 98.14, 96.66, 94.15, 89.66, 83.73, 76.46],
    'pgd10': [99.15, 98.59, 97.69, 96.46, 94.14, 91.01, 86.95],
    'pgd20': [99.13, 98.66, 98.16, 97.30, 96.30, 94.78, 93.38],
    'pgd40': [99.06, 98.60, 97.99, 97.36, 96.72, 95.60, 94.70],
}

STYLES = {
    'base':  {'label': 'Baseline (no defense)', 'color': '#E24B4A', 'linestyle': '--'},
    'fgsm':  {'label': 'FGSM defense (ε=0.20)', 'color': '#1D9E75', 'linestyle': '-'},
    'pgd5':  {'label': 'PGD5 defense (ε=0.15)', 'color': '#378ADD', 'linestyle': '-'},
    'pgd10': {'label': 'PGD10 defense (ε=0.15)', 'color': '#BA7517', 'linestyle': '-'},
    'pgd20': {'label': 'PGD20 defense (ε=0.15)', 'color': '#7F77DD', 'linestyle': '-'},
    'pgd40': {'label': 'PGD40 defense (ε=0.15)', 'color': '#F39C12', 'linestyle': '-'},
}

def main():
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_key, accuracies in RESULTS.items():
        if any(v is None for v in accuracies):
            continue

        style = STYLES[model_key]
        ax.plot(
            epsilons,
            accuracies,
            label=style['label'],
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=2,
            marker='o',
            markersize=5
        )

    ax.set_xlabel('Epsilon', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Defense comparison under PGD40 attack', fontsize=13)
    ax.set_xticks(epsilons)
    ax.set_ylim(0, 105)
    ax.axhline(y=10, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.annotate('random guessing (~10%)', xy=(0.01, 11), fontsize=9, color='gray')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {save_path}')

if __name__ == '__main__':
    main()