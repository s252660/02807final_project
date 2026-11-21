"""Plot Gibbs log-likelihood convergence for K=7 across random seeds."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "lda_grid_search"
OUT = ROOT / "data" / "lda_examples"
K = 7
SEEDS = range(1, 6)


def main():
    convergence = {}

    for seed in SEEDS:
        model_path = RUNS / f"k{K}_seed{seed}" / "model.npz"
        if not model_path.exists():
            continue

        with np.load(model_path) as model:
            convergence[seed] = np.asarray(model["log_likelihood"], dtype=float)

    if not convergence:
        raise FileNotFoundError(f"No K={K} seed runs found in {RUNS}")

    n_sweeps = min(len(values) for values in convergence.values())
    sweeps = np.arange(1, n_sweeps + 1)
    matrix = np.vstack([
        np.asarray(values[:n_sweeps], dtype=float)
        for values in convergence.values()
    ])
    mean = matrix.mean(axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for seed, values in convergence.items():
        for sweep, likelihood in zip(sweeps, values[:n_sweeps]):
            csv_rows.append({
                "K": K,
                "seed": seed,
                "sweep": int(sweep),
                "log_likelihood": float(likelihood),
            })
    pd.DataFrame(csv_rows).to_csv(OUT / "k7_gibbs_convergence.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0.1, 0.9, len(convergence)))

    for (seed, values), color in zip(convergence.items(), colors):
        ax.plot(
            sweeps,
            values[:n_sweeps],
            color=color,
            linewidth=1.2,
            alpha=0.55,
            label=f"Seed {seed}",
        )

    ax.plot(
        sweeps,
        mean,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="Mean across seeds",
    )

    ax.set_xlabel("Gibbs sweep", fontsize=11)
    ax.set_ylabel("Log likelihood log P(w | z)", fontsize=11)
    ax.set_title("Gibbs convergence for K=7", fontsize=14, weight="bold")
    ax.grid(axis="both", alpha=0.22)
    ax.legend(ncol=3, frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    plot_path = OUT / "k7_gibbs_convergence.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {plot_path}")
    print(f"Saved data: {OUT / 'k7_gibbs_convergence.csv'}")


if __name__ == "__main__":
    main()
