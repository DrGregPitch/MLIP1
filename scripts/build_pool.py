#!/usr/bin/env python3
"""Build a TAILED configuration pool and label it with cheap + high-fidelity MLIPs.

    python scripts/build_pool.py

Faithful to how real MLIP active-learning fine-tuning works: the pool spans mild ->
moderate distortion, so it has a long tail of configurations where the cheap base
model (MACE-OFF small) is genuinely wrong and the high-fidelity oracle (MACE-OFF
large; proxy for DFT) differs. The correction to be learned is concentrated in that
tail -- which is exactly where active learning should spend the oracle budget.

Caches data_cache/pool.npz with:
  X   features: Coulomb eigenvalues + base-model energy (prior) + base |F|max
  y   target: high-fidelity oracle energy (per-molecule-per-model centered)
  base_pred: the base model's centered energy = the no-fine-tuning baseline
  mol: molecule id
Absolute energies are centered per molecule per model to remove the cross-model
reference-energy offset; unphysical (atom-overlap) configs are filtered out.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlip1.core import coulomb_eigs  # noqa: E402

MOLECULES = ("CH3CH2OH", "CH3COOH", "CH3CN", "CH3CHO")


def make_tailed(n_per_mol=220, seed=0):
    from ase.build import molecule
    rng = np.random.default_rng(seed)
    atoms, mol = [], []
    for mi, name in enumerate(MOLECULES):
        base = molecule(name)
        for _ in range(n_per_mol):
            a = base.copy()
            a.rattle(rng.uniform(0.02, 0.28), seed=int(rng.integers(1e6)))  # mild->moderate tail
            atoms.append(a)
            mol.append(mi)
    return atoms, np.asarray(mol)


def main():
    from mace.calculators import mace_off
    out = Path("data_cache")
    out.mkdir(exist_ok=True)

    print("loading MACE-OFF small (base) + large (high-fidelity oracle)...")
    small = mace_off(model="small", device="cpu", default_dtype="float64")
    large = mace_off(model="large", device="cpu", default_dtype="float64")

    atoms, mol = make_tailed()
    print(f"scoring {len(atoms)} configs with both models (this is the slow part)...")
    ce, e_s, e_l, fmax = [], [], [], []
    for a in atoms:
        a.calc = small
        e_s.append(a.get_potential_energy())
        fmax.append(float(np.abs(a.get_forces()).max()))
        a.calc = large
        e_l.append(a.get_potential_energy())
        ce.append(coulomb_eigs(a))
    ce = np.asarray(ce)
    e_s = np.asarray(e_s)
    e_l = np.asarray(e_l)
    fmax = np.asarray(fmax)
    mol = np.asarray(mol)

    # filter unphysical (atom-overlap) configs: extreme base force or energy outliers
    keep = fmax < 40.0
    for mi in np.unique(mol):
        m = mol == mi
        med = np.median(e_l[m])
        mad = np.median(np.abs(e_l[m] - med)) + 1e-9
        keep &= ~(m & (np.abs(e_l - med) > 6 * mad))
    ce, e_s, e_l, fmax, mol = ce[keep], e_s[keep], e_l[keep], fmax[keep], mol[keep]
    print(f"  kept {keep.sum()} physical configs (dropped {(~keep).sum()})")

    # center per molecule per model -> removes the cross-model offset; the geometry-
    # dependent correction (tail) survives
    es_c = np.zeros_like(e_s)
    el_c = np.zeros_like(e_l)
    for mi in np.unique(mol):
        m = mol == mi
        es_c[m] = e_s[m] - e_s[m].mean()
        el_c[m] = e_l[m] - e_l[m].mean()

    gap = np.abs(el_c - es_c)
    print(f"  correction (|E_large - E_small|, centered): median {np.median(gap):.3f} eV, "
          f"90th pct {np.percentile(gap,90):.3f} eV  <- the tail")

    X = np.column_stack([ce, es_c, np.log1p(fmax)])   # descriptors + base prior + base instability
    np.savez(out / "pool.npz", X=X, y=el_c, base_pred=es_c, mol=mol, fmax=fmax)
    print(f"cached -> {out/'pool.npz'} | X {X.shape}, target std {el_c.std():.3f} eV")


if __name__ == "__main__":
    main()
