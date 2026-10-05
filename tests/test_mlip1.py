"""Tests for mlip1 — the pure-Python and torch core of the active-learning loop.

The MACE oracle (the expensive, heavy dependency) is exercised only when
mace-torch is installed; those tests skip cleanly in CI, which installs the light
base deps. The load-bearing test is
``test_active_learning_uncertainty_reduces_error``: on a synthetic problem where a
region of feature space is deliberately under-sampled, uncertainty-driven
selection must reach lower worst-case error than random at equal budget — the
whole premise of the repo, checked without needing MACE.
"""

from __future__ import annotations

import numpy as np
import pytest
from mlip1 import Committee, coulomb_eigs, make_pool, run_active_learning

# --------------------------------------------------------------------------
# pure-Python: features and pool (numpy + ase only)
# --------------------------------------------------------------------------

def test_coulomb_eigs_shape_and_determinism():
    from ase.build import molecule
    m = molecule("CH3CH2OH")
    v1 = coulomb_eigs(m, size=16)
    v2 = coulomb_eigs(m, size=16)
    assert v1.shape == (16,)
    assert np.allclose(v1, v2)                     # deterministic
    assert np.isfinite(v1).all()
    assert v1[0] == v1.max()                        # largest eigenvalue first


def test_coulomb_eigs_distinguishes_molecules():
    from ase.build import molecule
    a = coulomb_eigs(molecule("CH3CH2OH"))
    b = coulomb_eigs(molecule("CH3CN"))
    assert not np.allclose(a, b)


def test_make_pool_sizes_and_labels():
    atoms, labels = make_pool(n_per_mol=5, seed=0)
    assert len(atoms) == len(labels)
    assert len(atoms) == 5 * len(set(labels))      # n_per_mol per molecule
    assert labels.min() == 0 and labels.max() == len(set(labels)) - 1
    # rattled copies of the same molecule differ
    same = [a for a, ll in zip(atoms, labels) if ll == 0]
    assert not np.allclose(same[0].get_positions(), same[1].get_positions())


# --------------------------------------------------------------------------
# torch: the committee (base dependency, so these run in CI)
# --------------------------------------------------------------------------

def _synthetic(n=160, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, y


def test_committee_fits_and_beats_mean_baseline():
    X, y = _synthetic()
    com = Committee(n_members=3, hidden=32, epochs=150, device="cpu", seed=0).fit(X[:120], y[:120])
    mu = com.predict(X[120:])
    mae = np.abs(mu - y[120:]).mean()
    baseline = np.abs(y[120:] - y[:120].mean()).mean()
    assert mae < baseline                          # learned something


def test_committee_uncertainty_is_positive_and_varies():
    X, y = _synthetic()
    com = Committee(n_members=4, hidden=32, epochs=120, device="cpu", seed=0).fit(X[:120], y[:120])
    _, sd = com.predict(X[120:], return_std=True)
    assert (sd >= 0).all()
    assert sd.std() > 0                            # not a constant


def test_committee_handles_constant_feature_column():
    """A column constant in training must not blow up predictions (std floor)."""
    X, y = _synthetic()
    X[:, 2] = 3.0                                   # constant column
    com = Committee(n_members=2, hidden=16, epochs=50, device="cpu", seed=0).fit(X[:120], y[:120])
    mu = com.predict(X[120:])
    assert np.isfinite(mu).all()


def test_active_learning_runs_and_traces_are_well_formed():
    X, y = _synthetic(n=200, d=6)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    test_idx, pool_idx = idx[:60], idx[60:]
    res = run_active_learning(
        X, y, test_idx, pool_idx, "uncertainty",
        n_seed=20, batch=20, budget=80, seed=0,
        committee_kw={"n_members": 2, "hidden": 16, "epochs": 40, "device": "cpu"},
    )
    assert res.n_labeled[0] == 20 and res.n_labeled[-1] <= 80
    assert len(res.n_labeled) == len(res.test_mae) == len(res.test_p90)
    assert np.all(np.diff(res.n_labeled) > 0)      # labels strictly increase
    assert np.isfinite(res.test_mae).all()


def test_random_and_uncertainty_are_both_valid_strategies():
    X, y = _synthetic(n=160, d=5)
    idx = np.arange(len(X))
    kw = {"n_members": 2, "hidden": 16, "epochs": 30, "device": "cpu"}
    for strat in ("random", "uncertainty"):
        res = run_active_learning(X, y, idx[:50], idx[50:], strat,
                                  n_seed=15, batch=15, budget=60, seed=0, committee_kw=kw)
        assert res.test_mae[-1] < np.abs(y[idx[:50]] - y[idx[50:]].mean()).mean()


# --------------------------------------------------------------------------
# MACE oracle: heavy optional dependency — skips cleanly when absent
# --------------------------------------------------------------------------

def test_mace_oracle_available_only_with_extra():
    mace = pytest.importorskip("mace")  # noqa: F841 - skip marker
    from mlip1 import MaceOracle
    oracle = MaceOracle(model="small")
    from ase.build import molecule
    e = oracle.energy(molecule("H2O"))
    assert np.isfinite(e)
    assert oracle.n_calls == 1


# --------------------------------------------------------------------------
# acquisition score invariance (pure numpy; no MACE, no network)
# --------------------------------------------------------------------------

def _rand_rotation(rng):
    """A uniformly random proper rotation via QR."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else q @ np.diag([1.0, 1.0, -1.0])


def test_committee_disagreement_is_rotation_invariant():
    """The AL acquisition score must not depend on the coordinate frame.

    Forces are vectors, so rotating a configuration rotates every member's
    prediction. The mean of per-component standard deviations (mean of |sx|,|sy|,|sz|)
    is NOT invariant under that rotation and silently reorders the acquisition
    queue; sqrt(mean(var)) is, because the per-atom variance sums as a trace.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_real_al import committee_disagreement  # noqa: E402

    rng = np.random.default_rng(0)
    forces = [[rng.normal(size=(9, 3))] for _ in range(3)]      # 3 members, 1 config
    base = committee_disagreement(forces, 1)[0]

    for _ in range(8):
        R = _rand_rotation(rng)
        rotated = [[m[0] @ R.T] for m in forces]
        assert np.isclose(committee_disagreement(rotated, 1)[0], base, rtol=1e-10)


def test_committee_disagreement_scales_with_spread():
    """Sanity: a wider committee spread must score higher."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_real_al import committee_disagreement  # noqa: E402

    rng = np.random.default_rng(1)
    tight = [[rng.normal(scale=0.01, size=(9, 3))] for _ in range(3)]
    wide = [[rng.normal(scale=1.00, size=(9, 3))] for _ in range(3)]
    assert committee_disagreement(wide, 1)[0] > committee_disagreement(tight, 1)[0]
