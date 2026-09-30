#!/usr/bin/env python3
"""Build the pool + an OUT-OF-DISTRIBUTION test split -- the correct evaluation for
active-learning fine-tuning.

Faithful to real MD-based sampling: configuration difficulty is skewed toward
near-equilibrium (a Beta-distributed rattle amplitude), so the pool is mostly easy
configs with only a RARE hard tail -- exactly what a molecular-dynamics trajectory
visits. The held-out TEST set is drawn from the HARD tail (the deployment regime an
MLIP must be reliable on). Active learning should win here: its committee finds the
rare hard configs in the pool, while random mostly labels the easy bulk.

"Hardness" = the base model's force error vs. the high-fidelity reference (where the
universal MLIP is wrong). Unphysical near-overlap configs (ref |F|max >= 40 eV/A)
are filtered.
"""
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from mace.calculators import mace_off

MOLS = ("CH3CH2OH", "CH3COOH", "CH3CN")


def main():
    out = Path("data_cache")
    out.mkdir(exist_ok=True)
    small = mace_off(model="small", device="cpu", default_dtype="float64")
    large = mace_off(model="large", device="cpu", default_dtype="float64")
    rng = np.random.default_rng(0)

    kept, hardness = [], []
    for name in MOLS:
        base = molecule(name)
        for _ in range(180):
            a = base.copy()
            amp = 0.02 + 0.30 * rng.beta(1.5, 4.5)   # MD-like: mostly small, rare large
            a.rattle(amp, seed=int(rng.integers(1e6)))
            a.calc = large
            e = a.get_potential_energy()
            fl = a.get_forces()
            if np.abs(fl).max() >= 40.0:
                continue
            a.calc = small
            fs = a.get_forces()
            a.calc = SinglePointCalculator(a, energy=e, forces=fl)  # reference labels
            kept.append(a)
            hardness.append(np.sqrt(((fs - fl) ** 2).mean()))       # base error = hardness

    kept = np.array(kept, dtype=object)
    hardness = np.asarray(hardness)
    order = np.argsort(hardness)
    hard_cut = order[int(0.65 * len(order)):]     # top 35% hardest = the OOD regime
    rng.shuffle(hard_cut)
    test_idx = set(hard_cut[:50].tolist())        # 50 hard configs held out for test

    test = [kept[i] for i in range(len(kept)) if i in test_idx]
    pool = [kept[i] for i in range(len(kept)) if i not in test_idx]
    pool_hard = np.mean([hardness[i] > np.quantile(hardness, 0.65)
                         for i in range(len(kept)) if i not in test_idx])

    write(out / "ft_heldout.xyz", test)
    write(out / "ft_pool.xyz", pool)
    print(f"kept {len(kept)} physical configs")
    print(f"  TEST = {len(test)} HARD/OOD configs (base force error "
          f">{np.quantile(hardness,0.65)*1000:.0f} meV/A)")
    print(f"  POOL = {len(pool)} configs, only {pool_hard*100:.0f}% hard "
          f"-> random rarely labels the hard tail; AL should find it")


if __name__ == "__main__":
    main()
