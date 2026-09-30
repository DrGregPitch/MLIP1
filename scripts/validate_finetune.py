#!/usr/bin/env python3
"""Validate real MACE fine-tuning on HELD-OUT configs: does fine-tuning MACE-small
toward the reference reduce force error on configs it never trained on?

    python scripts/validate_finetune.py

Base MACE-small vs. fine-tuned MACE-small, force error against MACE-large
reference labels on the held-out test set.

Two review-driven corrections baked in:
* The fine-tune's early-stopping/validation split comes from the TRAIN file
  (``--valid_fraction``), never the test set -- selecting the checkpoint on the
  test set would optimistically bias the headline.
* The reported number is the **median per-config force RMSE over physical
  configs** (reference |F|max < 40 eV/A). Aggregate RMSE is dominated by a
  handful of near-atom-overlap outliers -- that is exactly the "136x" artifact
  this project already caught once.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read


def per_config_rmse(model_paths, test_atoms):
    """Force RMSE per config (meV/A) vs the baked-in reference labels."""
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=model_paths, device="cpu", default_dtype="float64")
    out = []
    for a in test_atoms:
        ref = a.get_forces()          # baked-in MACE-large reference
        b = a.copy()
        b.calc = calc
        err = b.get_forces() - ref
        out.append(np.sqrt((err ** 2).mean()) * 1000)
    return np.asarray(out)


def main():
    test = read("data_cache/ft_test.xyz", ":")
    physical = np.array([np.abs(a.get_forces()).max() < 40.0 for a in test])
    print(f"held-out configs: {len(test)} ({physical.sum()} physical)")

    fm = str(Path.home() / ".cache/mace/MACE-OFF23_small.model")

    print("base MACE-small per-config force RMSE on held-out test...")
    base = per_config_rmse(fm, test)

    print("\nfine-tuning MACE-small toward the reference (real mace_run_train)...")
    t0 = time.time()
    try:
        subprocess.run([
            ".venv/bin/mace_run_train", "--name=ft_val",
            f"--foundation_model={fm}",
            "--train_file=data_cache/ft_train.xyz",
            "--valid_fraction=0.1",            # validation from TRAIN, never test
            "--energy_key=energy", "--forces_key=forces", "--E0s=foundation",
            "--multiheads_finetuning=False",
            "--max_num_epochs=120", "--batch_size=10", "--lr=0.01",
            "--device=cpu", "--default_dtype=float64",
            "--model_dir=ft_work", "--log_dir=ft_work", "--results_dir=ft_work",
            "--checkpoints_dir=ft_work", "--seed=0",
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print("mace_run_train FAILED; stderr tail:\n", (exc.stderr or "")[-2000:])
        raise
    print(f"  fine-tuned in {time.time()-t0:.0f}s")

    ft = per_config_rmse("ft_work/ft_val.model", test)

    b_med = float(np.median(base[physical]))
    f_med = float(np.median(ft[physical]))
    print(f"\n  base       median force RMSE (physical configs): {b_med:.0f} meV/A")
    print(f"  fine-tuned median force RMSE (physical configs): {f_med:.0f} meV/A")
    if f_med < b_med:
        print(f"  -> fine-tuning cut held-out force error {b_med / f_med:.1f}x (median)")
    else:
        print("  -> no improvement")
    print(f"  (aggregate incl. outliers, for reference: {np.sqrt((base**2).mean()):.0f} "
          f"-> {np.sqrt((ft**2).mean()):.0f} meV/A)")


if __name__ == "__main__":
    sys.exit(main())
