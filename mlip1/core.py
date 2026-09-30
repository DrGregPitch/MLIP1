"""MLIP active-learning core: oracle, features, a GPU-trained committee, the loop.

The pattern (ALCHEMI's MLIP active learning for fine-tuning), on a locally-runnable
proxy:

* **Oracle** — an expensive atomistic reference. Here the MACE-OFF *medium*
  foundation model scores each configuration's energy; every call is counted. In
  production this is a GPU-DFT call, and the loop is unchanged.
* **Model** — a system-specific energy model trained on cheap descriptors. Here a
  small **committee of MLPs trained on Metal/MPS**; in production this is the MLIP
  being fine-tuned.
* **Acquisition** — committee disagreement. Label the configurations the committee
  is most uncertain about; retrain; repeat.

Claim to demonstrate: uncertainty-driven selection reaches a target accuracy in far
fewer oracle calls than random sampling.

Labels for a fixed candidate pool are computed once and cached, so the
active-learning benchmark (many restarts) is fast; an "oracle call" is then a
counted reveal of a cached label — standard practice for AL method benchmarking.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# configuration pool + descriptors
# --------------------------------------------------------------------------

MOLECULES = ("CH3CH2OH", "CH3OCH3", "CH3COOH", "CH3CHO", "CH3CN", "C2H6")


def make_pool(n_per_mol: int = 150, seed: int = 0):
    """Diverse conformers of several small molecules -> a real regression target.

    Multiple chemistries give active learning room to work: random sampling covers
    them unevenly by luck, while uncertainty-driven selection balances across the
    space. Rattle amplitudes are kept mild so configurations stay physical.
    """
    from ase.build import molecule

    rng = np.random.default_rng(seed)
    atoms, labels = [], []
    for mi, name in enumerate(MOLECULES):
        base = molecule(name)
        for _ in range(n_per_mol):
            a = base.copy()
            a.rattle(rng.uniform(0.02, 0.12), seed=int(rng.integers(1e6)))
            atoms.append(a)
            labels.append(mi)
    return atoms, np.asarray(labels)


def coulomb_eigs(atoms, size: int = 16) -> np.ndarray:
    """Sorted Coulomb-matrix eigenvalues: a cheap, geometry- and composition-aware
    descriptor. This is the 'cheap features' the system-specific model sees."""
    Z = atoms.get_atomic_numbers().astype(float)
    R = atoms.get_positions()
    n = len(Z)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = 0.5 * Z[i] ** 2.4 if i == j else Z[i] * Z[j] / (
                np.linalg.norm(R[i] - R[j]) + 1e-9
            )
    ev = np.sort(np.linalg.eigvalsh(M))[::-1]
    out = np.zeros(size)
    out[: min(size, n)] = ev[:size]
    return out


# --------------------------------------------------------------------------
# oracle: MACE-OFF medium, with a call counter
# --------------------------------------------------------------------------

class MaceOracle:
    """Wraps MACE-OFF medium as the expensive energy reference, counting calls."""

    def __init__(self, model: str = "medium"):
        from mace.calculators import mace_off

        self.calc = mace_off(model=model, device="cpu", default_dtype="float64")
        self.n_calls = 0

    def energy(self, atoms) -> float:
        self.n_calls += 1
        a = atoms.copy()
        a.calc = self.calc
        return float(a.get_potential_energy())


def label_pool(atoms_list) -> np.ndarray:
    """Compute (and return) the oracle energy for every configuration, once."""
    oracle = MaceOracle()
    return np.array([oracle.energy(a) for a in atoms_list])


# --------------------------------------------------------------------------
# committee of MLPs, trained on Metal (MPS)
# --------------------------------------------------------------------------

def pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Committee:
    """K small MLPs; mean = prediction, std across members = uncertainty.

    Trained in float32 on MPS (Metal) when available — the GPU-accelerated half of
    the workflow. Features and target are standardized internally.
    """

    n_members: int = 4
    hidden: int = 64
    epochs: int = 120
    lr: float = 5e-3
    device: str = "auto"
    seed: int = 0

    def _build(self, in_dim, torch, nn):
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, self.hidden), nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        dev = pick_device() if self.device == "auto" else self.device
        self._dev = dev
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        self._xm = X.mean(0)
        xs = X.std(0)
        # a column constant in the labeled subset but varying in the pool would
        # be divided by ~1e-8 at predict time (~1e8-scale inputs, silently
        # garbage predictions); treat near-constant columns as unscaled instead
        self._xs = np.where(xs < 1e-8, 1.0, xs)
        self._ym, self._ys = y.mean(), y.std() + 1e-8
        Xn = torch.as_tensor((X - self._xm) / self._xs, device=dev)
        yn = torch.as_tensor((y - self._ym) / self._ys, device=dev).view(-1, 1)

        rng = np.random.default_rng(self.seed)
        self._members = []
        for k in range(self.n_members):
            torch.manual_seed(self.seed + k)
            idx = rng.integers(0, len(X), len(X))  # bootstrap for diversity
            net = self._build(X.shape[1], torch, nn).to(dev)
            opt = torch.optim.Adam(net.parameters(), self.lr)
            xb, yb = Xn[idx], yn[idx]
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = ((net(xb) - yb) ** 2).mean()
                loss.backward()
                opt.step()
            self._members.append(net)
        return self

    def predict(self, X, return_std: bool = False):
        import torch

        Xn = torch.as_tensor(
            (np.asarray(X, np.float32) - self._xm) / self._xs, device=self._dev
        )
        with torch.no_grad():
            preds = np.stack([
                m(Xn).cpu().numpy().ravel() * self._ys + self._ym for m in self._members
            ])
        mu = preds.mean(0)
        if return_std:
            return mu, preds.std(0)
        return mu


# --------------------------------------------------------------------------
# the active-learning loop
# --------------------------------------------------------------------------

def _standardize(X_ref, X):
    """Standardize X by X_ref's per-column mean/std (for feature-space distances)."""
    mu, sd = X_ref.mean(0), X_ref.std(0) + 1e-8
    return (np.asarray(X, float) - mu) / sd


@dataclass
class ALResult:
    strategy: str
    n_labeled: np.ndarray   # oracle calls after each round
    test_mae: np.ndarray    # mean test error (eV) after each round
    test_p90: np.ndarray = None   # 90th-pctile (worst-case) test error -- the metric
    #                               that matters for MD stability: rare large errors
    #                               crash simulations, and killing them is why MLIP
    #                               active learning exists.


def run_active_learning(
    X, y, test_idx, pool_idx, strategy: str,
    n_seed: int = 12, batch: int = 6, budget: int = 120, seed: int = 0,
    committee_kw: dict | None = None,
) -> ALResult:
    """One active-learning campaign over a fixed, pre-labeled pool.

    ``strategy`` is ``"uncertainty"`` (query max committee-std) or ``"random"``.
    Returns the test-MAE trace vs. number of oracle labels used.
    """
    rng = np.random.default_rng(seed)
    committee_kw = committee_kw or {}
    pool = list(pool_idx)
    labeled = list(rng.choice(pool, size=n_seed, replace=False))
    remaining = [i for i in pool if i not in set(labeled)]

    n_lab, maes, p90s = [], [], []
    while True:
        com = Committee(seed=seed, **committee_kw).fit(X[labeled], y[labeled])
        err = np.abs(com.predict(X[test_idx]) - y[test_idx])
        maes.append(float(err.mean()))
        p90s.append(float(np.percentile(err, 90)))  # worst-case metric
        n_lab.append(len(labeled))
        if len(labeled) >= budget or not remaining:
            break

        take = min(batch, len(remaining))
        if strategy == "random":
            pick = list(rng.choice(remaining, size=take, replace=False))
        elif strategy in ("uncertainty", "uncertainty_diverse"):
            _, sigma = com.predict(X[remaining], return_std=True)
            if strategy == "uncertainty":
                order = np.argsort(sigma)[::-1][:take]
                pick = [remaining[i] for i in order]
            else:
                # diversity-gated: from the most-uncertain candidates, greedily
                # pick a batch that is spread out in feature space (farthest-point),
                # so we don't spend the budget on near-duplicate uncertain configs.
                cand = [remaining[i] for i in np.argsort(sigma)[::-1][: max(take * 4, take)]]
                Xc = _standardize(X, X[cand])
                Xl = _standardize(X, X[labeled])
                chosen: list[int] = []
                # distance of each candidate to the labeled set (novelty)
                mind = np.min(np.linalg.norm(Xc[:, None] - Xl[None], axis=2), axis=1)
                for _ in range(take):
                    j = int(np.argmax(mind))
                    chosen.append(cand[j])
                    dnew = np.linalg.norm(Xc - Xc[j], axis=1)
                    mind = np.minimum(mind, dnew)  # keep the batch spread out too
                    mind[j] = -1
                pick = chosen
        else:
            raise ValueError(strategy)
        labeled += pick
        rem = set(remaining) - set(pick)
        remaining = [i for i in remaining if i in rem]

    return ALResult(strategy, np.asarray(n_lab), np.asarray(maes), np.asarray(p90s))
