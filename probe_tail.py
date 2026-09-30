"""Premise probe: does the cheap-vs-high-fidelity MLIP gap have a LEARNABLE TAIL?

Real MLIP active-learning fine-tuning works because MD exploration produces
out-of-distribution configurations where the base model is genuinely wrong, and the
DFT-vs-model correction is concentrated there. My first attempt used only
near-equilibrium configs (no tail) -> nothing for AL to find -> random ties.

Test the premise honestly: sample configs across a WIDE distortion range (mild ->
severe), score with MACE-OFF small (cheap base) and large (high-fidelity proxy for
DFT), and check whether |E_large - E_small|:
  (1) GROWS in the distorted/high-energy tail (a real correctable gap), and
  (2) is PREDICTABLE from the base model's own uncertainty signal / features
      (so active learning can find those configs).

If both hold, rebuild the loop with a tailed pool. If not, say so.
"""

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from ase.build import molecule  # noqa: E402


def make_tailed(n=300, seed=0):
    """Configs across a wide distortion range -> a physical long tail."""
    rng = np.random.default_rng(seed)
    out, amp = [], []
    for name in ("CH3CH2OH", "CH3COOH", "CH3CN"):
        base = molecule(name)
        for _ in range(n // 3):
            a = base.copy()
            amplitude = rng.uniform(0.02, 0.55)  # WIDE: mild equilibrium -> severe distortion
            a.rattle(amplitude, seed=int(rng.integers(1e6)))
            out.append(a)
            amp.append(amplitude)
    return out, np.asarray(amp)


def main():
    from mace.calculators import mace_off
    from scipy.stats import spearmanr

    print("loading MACE-OFF small (base) + large (high-fidelity oracle)...")
    t0 = time.time()
    small = mace_off(model="small", device="cpu", default_dtype="float64")
    large = mace_off(model="large", device="cpu", default_dtype="float64")
    print(f"  ready in {time.time()-t0:.0f}s")

    configs, amp = make_tailed(n=300, seed=0)
    es, el, fmax_small = [], [], []
    t1 = time.time()
    for a in configs:
        a.calc = small
        es.append(a.get_potential_energy())
        fmax_small.append(np.abs(a.get_forces()).max())  # base-model force magnitude
        a.calc = large
        el.append(a.get_potential_energy())
    es, el = np.asarray(es), np.asarray(el)
    fmax_small = np.asarray(fmax_small)
    gap = np.abs(el - es)  # the correction fine-tuning must learn, per config
    print(f"  scored {len(configs)} configs in {time.time()-t1:.0f}s")

    # (1) does the gap grow in the tail?
    q = np.quantile(amp, [0.33, 0.66])
    lo = gap[amp <= q[0]].mean()
    mid = gap[(amp > q[0]) & (amp <= q[1])].mean()
    hi = gap[amp > q[1]].mean()
    print("\n(1) GAP vs DISTORTION (|E_large - E_small|, eV):")
    print(f"    mild {lo:.3f}  |  medium {mid:.3f}  |  severe {hi:.3f}   "
          f"-> tail is {hi/max(lo,1e-6):.0f}x the bulk")

    # (2) is the gap findable from a cheap signal the base model exposes?
    #     (force magnitude is a classic cheap novelty/instability proxy)
    rho_amp = spearmanr(amp, gap).statistic
    rho_f = spearmanr(fmax_small, gap).statistic
    print("\n(2) IS THE GAP FINDABLE?")
    print(f"    Spearman(distortion, gap)      = {rho_amp:.2f}")
    print(f"    Spearman(base |F|max, gap)     = {rho_f:.2f}")

    good = hi > 3 * lo and max(rho_amp, rho_f) > 0.4
    print("\nVERDICT:", "GO — real tail + findable; rebuild the loop with a tailed pool"
          if good else "the gap has no exploitable tail; AL genuinely won't help here")


if __name__ == "__main__":
    main()
