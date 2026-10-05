#!/usr/bin/env python3
"""Merge active-learning trace files and regenerate the money plot + summary.

    python scripts/plot_traces.py results_ood/real_al_traces.json \
        results_ood2/real_al_traces.json --out assets/mlip1_money_plot.png

Exists so the published figure and the README's improvement numbers are
reproducible from the raw traces by a committed script, instead of a manual
merge. Accepts any number of trace JSONs (e.g. runs extended via --seed-start)
and concatenates their campaigns per strategy.
"""
import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+", help="real_al_traces.json files to merge")
    p.add_argument("--out", default="assets/mlip1_money_plot.png")
    args = p.parse_args()

    merged = {"random": [], "uncertainty": []}
    for path in args.traces:
        d = json.loads(open(path).read())
        for k in merged:
            merged[k].extend(d.get(k, []))
    n = {k: len(v) for k, v in merged.items()}
    print(f"merged campaigns: {n}")
    grid = merged["random"][0]["n_labeled"]

    # every campaign must share the budget grid. A campaign that ended early via the
    # `not remaining` branch has a shorter trace, and np.array over ragged rows gives
    # an object array that silently mislabels the x-axis rather than failing.
    for k, campaigns in merged.items():
        for i, c in enumerate(campaigns):
            if c["n_labeled"] != grid:
                raise SystemExit(
                    f"{k} campaign {i} has budget grid {c['n_labeled']}, expected {grid}; "
                    "traces from different budget settings cannot be merged"
                )

    print("\nmean p90 (worst-case) force error, meV/A:")
    R = np.array([c["p90_rmse"] for c in merged["random"]])
    U = np.array([c["p90_rmse"] for c in merged["uncertainty"]])
    print("  random     ", "  ".join(f"{g}:{v:.0f}" for g, v in zip(grid, R.mean(0))))
    print("  active-learn", "  ".join(f"{g}:{v:.0f}" for g, v in zip(grid, U.mean(0))))
    imp = (R.mean(0) - U.mean(0)) / R.mean(0) * 100
    print("  AL better by:", "  ".join(f"{g}:{v:+.0f}%" for g, v in zip(grid, imp)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for arr, c, lab in [(R, "#888888", "random selection"),
                        (U, "#009e73", "active learning (committee uncertainty)")]:
        m, se = arr.mean(0), arr.std(0) / np.sqrt(len(arr))
        ax.plot(grid, m, color=c, lw=2.5, marker="o", ms=4, label=lab)
        ax.fill_between(grid, m - se, m + se, color=c, alpha=0.2)
    ax.set_xlabel("configurations labeled (reference calls) + fine-tuned on")
    ax.set_ylabel("worst-case (p90) force error on OOD test (meV/A)")
    ax.set_title(f"Real-MACE active-learning fine-tuning ({len(R)} restarts)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {args.out}")


if __name__ == "__main__":
    main()
