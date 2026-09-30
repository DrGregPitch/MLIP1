#!/usr/bin/env python3
"""Run the MLIP active-learning benchmark and draw the money plot.

    python scripts/build_pool.py     # once: label the pool with the MACE oracle
    python scripts/run_al.py --outdir results

Compares uncertainty-driven active learning against random sampling over many
restarts: test-set energy MAE vs. number of oracle calls (labels). The committee is
trained on Metal/MPS.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlip1 import pick_device, run_active_learning  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results", type=Path)
    p.add_argument("--restarts", default=16, type=int)
    p.add_argument("--budget", default=80, type=int)
    p.add_argument("--n-test", default=200, type=int)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()

    d = np.load("data_cache/pool.npz")
    X, y, mol = d["X"], d["y"], d["mol"]
    args.outdir.mkdir(exist_ok=True)
    print(f"pool {X.shape} | {len(np.unique(mol))} molecules | device {pick_device()}")
    print(f"budget {args.budget} of ~{len(X)-args.n_test} pool configs | "
          f"{args.restarts} restarts")

    strategies = ("random", "uncertainty")
    traces = {s: [] for s in strategies}
    grid = None
    t0 = time.time()
    for r in range(args.restarts):
        rng = np.random.default_rng(r)
        idx = rng.permutation(len(X))
        test_idx, pool_idx = idx[: args.n_test], idx[args.n_test:]
        for s in strategies:
            res = run_active_learning(
                X, y, test_idx, pool_idx, s, seed=r, budget=args.budget,
            )
            traces[s].append(res.test_mae)
            grid = res.n_labeled
        print(f"  restart {r+1}/{args.restarts} done ({time.time()-t0:.0f}s)")

    summary = {s: np.vstack(traces[s]) for s in strategies}
    np.savez(args.outdir / "al_traces.npz", grid=grid, **summary)

    # headline: labels to reach a target MAE (median over restarts)
    target = float(np.median([summary["random"][:, -1].mean(),
                              summary["uncertainty"][:, -1].mean()]) * 1.15)
    print(f"\nOracle calls to reach test MAE <= {target:.3f} eV (median over restarts):")
    calls = {}
    for s in strategies:
        hit = []
        for row in summary[s]:
            j = np.argmax(row <= target)
            hit.append(grid[j] if row[j] <= target else grid[-1] + 1)
        calls[s] = float(np.median(hit))
        shown = f"{calls[s]:.0f}" if calls[s] <= grid[-1] else f">{grid[-1]}"
        print(f"  {s:12s} {shown}")
    if calls["uncertainty"] < calls["random"]:
        print(f"\nActive learning: {calls['random']:.0f} -> {calls['uncertainty']:.0f} "
              f"oracle calls ({calls['random']/max(calls['uncertainty'],1):.2f}x fewer).")

    if args.no_figure:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"random": "#888888", "uncertainty": "#009e73"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for s in strategies:
        m = summary[s].mean(0)
        lo, hi = np.percentile(summary[s], 25, 0), np.percentile(summary[s], 75, 0)
        label = "active learning (committee uncertainty)" if s == "uncertainty" else "random sampling"
        ax.plot(grid, m, color=colors[s], lw=2, label=label)
        ax.fill_between(grid, lo, hi, color=colors[s], alpha=0.15)
    ax.axhline(target, color="crimson", ls="--", lw=1, label=f"target MAE ({target:.2f} eV)")
    ax.set_xlabel("oracle calls (configurations labeled by MACE)")
    ax.set_ylabel("test-set energy MAE (eV)")
    ax.set_title("MLIP active learning: fewer oracle calls to target accuracy\n"
                 "(MACE-OFF oracle, committee trained on Metal/MPS)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.outdir / "mlip1_money_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {args.outdir/'mlip1_money_plot.png'}")


if __name__ == "__main__":
    main()
