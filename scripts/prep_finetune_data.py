#!/usr/bin/env python3
"""Prepare a small real MACE fine-tuning dataset (energies + forces) and report the
cached foundation-model path. Used to measure real-fine-tune wall-clock cost.
"""
from pathlib import Path

import numpy as np
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from mace.calculators import mace_off


def main():
    out = Path("data_cache")
    out.mkdir(exist_ok=True)
    ref = mace_off(model="large", device="cpu", default_dtype="float64")

    rng = np.random.default_rng(0)
    configs = []
    for name in ("CH3CH2OH", "CH3COOH"):
        base = molecule(name)
        for _ in range(50):
            a = base.copy()
            a.rattle(rng.uniform(0.02, 0.25), seed=int(rng.integers(1e6)))
            a.calc = ref
            e = a.get_potential_energy()
            f = a.get_forces()
            a.calc = SinglePointCalculator(a, energy=e, forces=f)  # bake labels in
            configs.append(a)

    rng.shuffle(configs)
    train, test = configs[:80], configs[80:]
    write(out / "ft_train.xyz", train)
    write(out / "ft_test.xyz", test)
    print(f"wrote {len(train)} train / {len(test)} test configs (energy+forces from MACE-large)")

    cache = Path.home() / ".cache" / "mace" / "MACE-OFF23_small.model"
    print("foundation model (MACE-OFF small):", cache, "| exists:", cache.exists())


if __name__ == "__main__":
    main()
