#!/usr/bin/env python3
"""Relabel the pool with real DFT (PBE0/def2-SVP, PySCF) and rebuild the OOD split.

build_ft_pool.py labels with MACE-OFF *large* standing in for a DFT oracle. That has
two costs, and the second was only visible once real DFT was run on the same configs:

  * CORRELATED INDUCTIVE BIAS. Student and oracle are the same architecture family
    trained on the same distribution, so the student chases a target it is unusually
    suited to represent. Against real DFT there is a residual floor it cannot reach.
    The bias direction inflates the measured active-learning advantage.
  * THE ORACLE SATURATES, AND THAT BREAKS THE FILTER. MACE-OFF is trained on SPICE
    (near-equilibrium MD at 300-500 K) and has never seen a 0.66 A bond. On severely
    rattled geometries it does not extrapolate up the repulsive wall, it flattens:
    measured on 20 configurations, real DFT reaches 40-80 eV/A where the MACE label
    reports 10-23. build_ft_pool.py filters unphysical configs with
    |F|max >= 40 eV/A computed FROM THAT LABEL, so nothing in the pool ever reaches
    the threshold (observed maximum 27.2 eV/A) and near-dissociated geometries pass
    straight through. Half the held-out test set carries a sub-0.9 A bond.

Relabelling with DFT fixes both: the reference is uncorrelated with the student, and
the force filter can finally see what it was written to exclude.

The configurations themselves are NOT regenerated -- the same 536 structures are
reused, so only the labels change and the comparison against the previous run is
controlled.

ENERGY REFERENCE. PySCF returns total electronic energies (ethanol near -4190 eV),
on a different absolute scale from the foundation model's atomic references. Energies
are centred per molecule, and the AL driver must then be run with --e0s average.
Forces -- the reported metric -- are invariant to that shift.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.units import Bohr, Hartree

XC = "PBE0"
BASIS = "def2-SVP"
F_MAX = 40.0        # eV/A; now applied to REAL DFT forces, not a saturating model's
HARD_Q = 0.65       # top 35% by hardness = the OOD regime
TEST_N = 50


def dft_label(atoms):
    """-> (energy eV, forces eV/A) at XC/BASIS, or None if the SCF does not converge."""
    from pyscf import dft, gto
    m = gto.M(atom=[(s, tuple(p)) for s, p in
                    zip(atoms.get_chemical_symbols(), atoms.get_positions())],
              basis=BASIS, verbose=0)
    mf = dft.RKS(m)
    mf.xc = XC
    mf.conv_tol = 1e-9
    e = mf.kernel()
    if not mf.converged:
        return None
    f = -mf.nuc_grad_method().kernel() * (Hartree / Bohr)
    return float(e) * Hartree, f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="data_cache", type=Path)
    p.add_argument("--cache", default="data_cache/dft_labels.npz", type=Path)
    args = p.parse_args()

    cfgs = read("data_cache/ft_pool.xyz", ":") + read("data_cache/ft_heldout.xyz", ":")
    print(f"relabelling {len(cfgs)} configurations at {XC}/{BASIS}", flush=True)

    # resume from cache: a 36-minute job should not restart from zero
    done = {}
    if args.cache.exists():
        z = np.load(args.cache, allow_pickle=True)
        done = {int(k): v for k, v in zip(z["idx"], z["payload"])}
        print(f"  resuming: {len(done)} already labelled", flush=True)

    t0 = time.time()
    for i, a in enumerate(cfgs):
        if i in done:
            continue
        r = dft_label(a)
        done[i] = None if r is None else (r[0], r[1])
        if (i + 1) % 25 == 0 or i + 1 == len(cfgs):
            el = time.time() - t0
            n = sum(1 for k in done if k <= i)
            print(f"  {i+1:4d}/{len(cfgs)}  {el/60:5.1f} min elapsed  "
                  f"~{el/max(n,1)*(len(cfgs)-i-1)/60:5.1f} min left", flush=True)
            np.savez(args.cache, idx=np.array(list(done)),
                     payload=np.array(list(done.values()), dtype=object))
    np.savez(args.cache, idx=np.array(list(done)),
             payload=np.array(list(done.values()), dtype=object))

    failed = [i for i, v in done.items() if v is None]
    print(f"\nSCF converged {len(cfgs)-len(failed)}/{len(cfgs)}; {len(failed)} failures")

    # attach DFT labels, centre energies per molecule, apply the filter that can now see
    kept, dropped = [], 0
    by_formula: dict[str, list] = {}
    for i, a in enumerate(cfgs):
        v = done.get(i)
        if v is None:
            continue
        e, f = v
        if np.abs(f).max() >= F_MAX:
            dropped += 1
            continue
        b = a.copy()
        b.info["_e"] = e
        b.info["_f"] = f
        by_formula.setdefault(b.get_chemical_formula(), []).append(b)
        kept.append(b)
    for group in by_formula.values():
        mean = float(np.mean([g.info["_e"] for g in group]))
        for g in group:
            g.calc = SinglePointCalculator(g, energy=g.info["_e"] - mean, forces=g.info["_f"])
            del g.info["_e"], g.info["_f"]

    print(f"force filter |F|max >= {F_MAX} eV/A removed {dropped} of "
          f"{len(cfgs)-len(failed)} configs ({100*dropped/max(len(cfgs)-len(failed),1):.1f}%)")
    print("  -- the MACE-labelled pool never triggered this filter at all")

    # hardness = where the zero-shot foundation model is wrong vs real DFT
    from mace.calculators import mace_off
    small = mace_off(model="small", device="cpu", default_dtype="float64")
    hard = []
    for a in kept:
        b = a.copy()
        b.calc = small
        hard.append(np.sqrt(((b.get_forces() - a.get_forces()) ** 2).mean()))
    hard = np.asarray(hard)
    cut = np.quantile(hard, HARD_Q)

    rng = np.random.default_rng(0)
    tail = np.flatnonzero(hard > cut)
    rng.shuffle(tail)
    test_idx = set(tail[:TEST_N].tolist())
    test = [kept[i] for i in range(len(kept)) if i in test_idx]
    pool = [kept[i] for i in range(len(kept)) if i not in test_idx]

    write(args.outdir / "dft_heldout.xyz", test)
    write(args.outdir / "dft_pool.xyz", pool)
    print(f"\nreference: {XC}/{BASIS} (PySCF)")
    print(f"  zero-shot MACE-OFF small error vs DFT: median {np.median(hard)*1000:.0f}  "
          f"p90 {np.percentile(hard,90)*1000:.0f} meV/A")
    print(f"  TEST = {len(test)} hard/OOD configs (base error > {cut*1000:.0f} meV/A)")
    print(f"  POOL = {len(pool)} configs")
    print(f"\nrun: python scripts/run_real_al.py --pool {args.outdir}/dft_pool.xyz "
          f"--test {args.outdir}/dft_heldout.xyz --e0s average --outdir results_dft")


if __name__ == "__main__":
    main()
