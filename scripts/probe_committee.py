"""Precondition check for real-MACE active learning, before the multi-hour run.

Fine-tune a proper 3-member MACE committee on a modest set, then on held-out configs
ask: does committee force DISAGREEMENT correlate with committee force ERROR? That
correlation is the whole basis of active learning -- if high disagreement flags
high-error configs, selecting them helps; if not, AL can't beat random no matter the
scale.
"""
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_al import (  # noqa: E402
    committee_disagreement,
    committee_forces,
    finetune,
    per_config_force_rmse,
)


def main():
    from scipy.stats import spearmanr
    pool = read("data_cache/ft_pool.xyz", ":")
    test = read("data_cache/ft_heldout.xyz", ":")
    rng = np.random.default_rng(0)
    train = [pool[i] for i in rng.choice(len(pool), 40, replace=False)]

    print("fine-tuning a 3-member committee (80 epochs each)...")
    t0 = time.time()
    models = []
    for m in range(3):
        bt = list(rng.integers(0, len(train), len(train)))
        mp = finetune([train[i] for i in bt], f"probe_m{m}", "probe_work", 80, m)
        models.append(mp)
    print(f"  committee ready in {(time.time()-t0)/60:.1f} min")

    pf = committee_forces(models, test)
    err = per_config_force_rmse(pf, test)              # committee-mean force error
    dis = committee_disagreement(pf, len(test))        # committee disagreement (the AL signal)
    rho = spearmanr(dis, err).statistic
    print(f"\n  committee force error: median {np.median(err):.0f} meV/A")
    print(f"  disagreement vs error Spearman = {rho:.2f}")
    print("\nVERDICT:", "GO — disagreement is informative; launch the full AL run"
          if rho > 0.3 else "WEAK — committee uncertainty won't drive AL; don't burn hours")


if __name__ == "__main__":
    main()
