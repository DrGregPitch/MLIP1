"""De-risk probe v3 — the honest, DFT-less framing.

v2 finding: MACE-small already reproduces MACE-medium relative energies to ~0.08 eV
(the foundation models agree), so a small->medium "fine-tune" has nothing to learn,
and without local DFT I can't manufacture a large universal-model error.

Faithful reframe: demonstrate the ACTIVE-LEARNING SELECTION LOOP itself. Train a
system-specific energy model FROM CHEAP DESCRIPTORS to reproduce an expensive
atomistic oracle (MACE-medium; in production, DFT), and let committee uncertainty
choose which configs to label. This has real, reducible error -> a learning curve AL
can beat random on. It's the same loop ALCHEMI uses to fine-tune MLIPs against DFT,
with oracle and trainable-model swapped for locally-runnable stand-ins.

Verify: (A) a real learning curve (test MAE drops as labels increase, and starts
well above noise so there's headroom); (B) uncertainty tracks error.
"""

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from ase.build import molecule  # noqa: E402


def make_configs(n=400, seed=0):
    """Diverse conformers of a few small molecules -> a real regression target."""
    rng = np.random.default_rng(seed)
    out = []
    for name in ("CH3CH2OH", "CH3OCH3", "CH3COOH"):
        base = molecule(name)
        for _ in range(n // 3):
            a = base.copy()
            a.rattle(rng.uniform(0.03, 0.16), seed=int(rng.integers(1e6)))
            out.append(a)
    return out


def coulomb_eigs(atoms, size=16):
    Z = atoms.get_atomic_numbers().astype(float)
    R = atoms.get_positions()
    n = len(Z)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = 0.5 * Z[i] ** 2.4 if i == j else Z[i] * Z[j] / (np.linalg.norm(R[i] - R[j]) + 1e-9)
    ev = np.sort(np.linalg.eigvalsh(M))[::-1]
    out = np.zeros(size)
    out[: min(size, n)] = ev[:size]
    return out


def main():
    from mace.calculators import mace_off
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingRegressor

    print("loading MACE-OFF medium (expensive oracle)...")
    t0 = time.time()
    oracle = mace_off(model="medium", device="cpu", default_dtype="float64")
    print(f"  ready in {time.time()-t0:.0f}s")

    configs = make_configs(n=400, seed=0)
    X, e = [], []
    t1 = time.time()
    for a in configs:
        a.calc = oracle
        e.append(a.get_potential_energy())
        X.append(coulomb_eigs(a))
    X, e = np.asarray(X), np.asarray(e)
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-9
    keep = np.abs(e - med) < 8 * mad
    X, e = X[keep], e[keep]
    y = e - e.mean()
    print(f"  {keep.sum()} configs labeled in {time.time()-t1:.0f}s | "
          f"target std {y.std():.3f} eV")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    te = idx[-120:]
    pool = idx[:-120]

    def fit_committee(train_idx):
        c = []
        for k in range(5):
            b = rng.integers(0, len(train_idx), len(train_idx))
            m = GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=k)
            m.fit(X[train_idx][b], y[train_idx][b])
            c.append(m)
        return c

    print("\n(A) LEARNING CURVE (random subsets):")
    for nlab in (20, 40, 80, 160):
        c = fit_committee(pool[:nlab])
        p = np.stack([m.predict(X[te]) for m in c]).mean(0)
        print(f"    {nlab:3d} labels -> test MAE {np.abs(p - y[te]).mean():.3f} eV")

    # uncertainty vs error at a small label budget (where AL matters)
    c = fit_committee(pool[:40])
    preds = np.stack([m.predict(X[te]) for m in c])
    mu, sigma = preds.mean(0), preds.std(0)
    rho = spearmanr(sigma, np.abs(mu - y[te])).statistic
    print(f"\n(B) UNCERTAINTY TRACKS ERROR (@40 labels): Spearman = {rho:.2f}")

    mae20 = np.abs(np.stack([m.predict(X[te]) for m in fit_committee(pool[:20])]).mean(0) - y[te]).mean()
    mae160 = np.abs(np.stack([m.predict(X[te]) for m in fit_committee(pool[:160])]).mean(0) - y[te]).mean()
    good = mae20 > 1.3 * mae160 and rho > 0.2
    print("\nVERDICT:", "GO — real curve + informative uncertainty; build the AL loop"
          if good else "ADJUST")


if __name__ == "__main__":
    main()
