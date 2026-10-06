# Project 4 — MLIPs, GPU atomistics, ALCHEMI

**Status: public at github.com/DrGregPitch/MLIP1.**

The target role's core is **NVIDIA ALCHEMI**: MLIP-accelerated atomistic simulation
(geometry relaxation, conformer search, MD, TS search), **MLIP active learning for
automated fine-tuning**, and GNN property prediction — shipped as NIM microservices
+ Toolkit-Ops (GPU kernels) + the PyTorch-native ALCHEMI Toolkit. This project builds
toward that, prioritizing the parts that (a) run on this Mac and (b) reuse the
active-learning edge already demonstrated in `formulate`.

## The flag to plant (why this, for me specifically)

ALCHEMI does **MLIP active learning for automated fine-tuning** — a loop that picks
*which* structures to run expensive DFT on, to most improve the potential. That is
the *same* surrogate + acquisition pattern as `formulate`, pointed at a new target.
It's the one ALCHEMI capability where I start from genuine strength. **That is the
headline deliverable.** Everything before it is the hands-on foundation the role's
brief lays out.

## Two concepts to demonstrate, not just cite

1. **Conservative vs. non-conservative forces.** Forces as the energy gradient
   (∂E/∂x) vs. predicted by a separate head. Non-conservative is faster but breaks
   energy conservation, so MD/relaxation can go unstable — e.g. orb-d3-v2
   (non-conservative) accurately optimized only ~39% of MOFs vs. ~94% for a
   gradient-based Orb. A clean local experiment: relax/MD the same structures with a
   conservative vs. non-conservative potential and show the stability difference.
2. **Fine-tuning beats universality.** A universal MLIP is triage; fine-tuning on the
   property you care about improves forces 5–15× and energy by orders of magnitude.
   The active-learning loop above is the *efficient* way to choose the fine-tuning
   set.

## Build order (the brief's hands-on path, scoped to runnable-here)

| step | what | runs on this Mac? |
|---|---|---|
| 1 | `mace-torch` + ASE; relax a structure, run MD, compute a phonon/energy curve | ✅ CPU/MPS |
| 2 | Swap **Orb** and (gated) **UMA** behind the same ASE calculator; compare energy/forces/speed, and conservative vs non-conservative | ✅ Orb/CPU; UMA needs HF license |
| 3 | **MLIP active-learning fine-tuning loop** — GP/ensemble uncertainty picks structures → "label" with a reference → fine-tune MACE → repeat; show fewer labels reach target force accuracy vs random | ✅ small-scale CPU/MPS |
| 4 | Read `NVIDIA/nvalchemi-toolkit-ops` (warp-lang kernels: Velocity Verlet, Langevin, Nosé-Hoover, FIRE, Ewald/PME); note what a batched-relaxation NIM does | ⚠️ read-only (CUDA) / cloud GPU |

## GPU-DFT context (for the "where's the training data from?" question)

The reference labels an MLIP fine-tunes against come from DFT. GPU-accelerated DFT is
where that's generated at scale: **GPU4PySCF, TeraChem, QUICK**, and GPU builds of
**VASP/CP2K**. Locally I can't run these, but the fine-tuning loop is written so the
"oracle" is pluggable. It now runs against real DFT locally (PBE0/def2-SVP via PySCF,
36 min for the whole pool — these molecules are 6–9 atoms); at production scale the
same slot takes a GPU-DFT call.

## Talking points this earns (honestly)

MLIP · foundation models for atomistic simulation (MACE, Orb, UMA) · conservative vs
non-conservative forces and MD stability · MLIP active-learning fine-tuning (the
ALCHEMI pattern) · the ALCHEMI three-layer architecture and where GPU-DFT fits.

## Device strategy on this Mac (Apple Silicon, 96 GB, Metal/MPS)

Measured, not assumed:

- **MLIP simulation (MACE relax/MD via ASE calculator) → CPU.** MACE runs float64
  internally; **MPS has no float64 support at all**, so the calculator can't go on
  Metal without patching MACE's core. CPU is fine here (~74 ms / 192-atom energy),
  and 96 GB RAM means large systems fit.
- **ML training (fine-tuning, GNN property head — the flag) → Metal/MPS.** float32
  training benchmarks **~6× faster than CPU** (4.8 vs 28 ms/step), and unified
  memory avoids host↔device copies.
- **Interview point:** the float64 gap is a real reason the production atomistics
  stack (ALCHEMI toolkit-ops, warp-lang) is **CUDA**-native — CUDA supports float64,
  Metal doesn't.
- **Orb needs a Python 3.12 venv** (its `dm-tree` dep has no 3.13 wheel). A
  packaging issue, unrelated to GPU.

## CORRECTION (measured): real MACE fine-tuning IS feasible locally

Earlier I wrongly claimed fine-tuning the real MLIP needed a CUDA GPU. Two errors:
float64 works fine on the Mac **CPU** (only the Metal GPU lacks it), and I
overestimated cost. Measured: a real `mace_run_train --foundation_model` fine-tune
is **~1.5 s/epoch on CPU** (80 configs) -> a full fine-tune is 2-3 min, and force
RMSE drops from ~147000 to ~2000 meV/A in 3 epochs.

**Implication:** the faithful loop uses REAL MACE fine-tuning + a committee of
fine-tuned MACEs (reducible uncertainty), run locally in the background. My earlier
toy-surrogate null (AL not beating random) was because the surrogate's tail error
was *irreducible* (representational); the real high-capacity MLIP fixes that. MACE
ships the exact tools: `mace_run_train --foundation_model`, `mace_finetuning_select`,
`mace_active_learning_md`.

## The crux, measured: surrogate vs. real MLIP uncertainty

The whole reason my first attempt failed and the real one should work, in one number
— committee force-disagreement vs. actual error (the precondition for AL > random):

| committee | disagreement-vs-error Spearman | AL beats random? |
|---|---|---|
| lightweight surrogate (MLP on Coulomb features) | 0.38 | no (irreducible tail error) |
| **real fine-tuned MACE committee** | **0.86** | yes (worst-case/OOD, +35-62% over 6 restarts) |

The high-capacity real MLIP has *reducible* uncertainty — it genuinely knows where
it is wrong — which is exactly what NVIDIA/ALCHEMI exploits and what a hand-feature
surrogate cannot replicate. Honest, hard-won, and the answer to "they know something
we don't": fine-tune the real model.

## Robust-number discipline (lessons that shaped this build)

Every too-good number this project produced was an artifact, caught and discarded:
cross-model absolute-energy offsets (10^11 eV), no-tail equilibrium pools, average
MAE hiding the worst-case metric that matters, and a "136x" force improvement that
was 4 atom-overlap outliers (robust median: ~2x). Report medians on physical
configs; distrust anything beating the domain literature's ballpark.

## RESULT (faithful, real MACE): active learning halves worst-case OOD error

The full arc, honestly, and what it took to earn a real win:

| setup | AL vs random |
|---|---|
| lightweight surrogate, any test | AL loses (uncertainty irreducible; Spearman 0.38) |
| real MACE committee, i.i.d. test, mean metric | AL loses (random near-optimal on a representative test) |
| **real MACE committee, OOD test, worst-case (p90) metric** | **AL wins +35-62% (6 restarts)** |

On an out-of-distribution test (the hard/deployment regime real MD visits), a committee
of fine-tuned MACE models selecting by force disagreement cut **worst-case (p90) force
error by 35-62%** vs random selection at equal reference-call budget, growing with
budget, over **6 restarts** with tight standard-error bands
(results_ood/money_plot_6restarts.png). Also established: real MACE fine-tuning cut
held-out force error ~3x (1121 -> 374 meV/A median, physical configs, train-side
validation split); committee disagreement tracks error at Spearman 0.86.

**The three conditions AL needs, learned the hard way:** (1) a high-capacity model
(real MLIP, not a surrogate) so uncertainty is reducible; (2) evaluate on the
deployment/OOD regime, not an i.i.d. test where random is near-optimal; (3) the
worst-case metric (MD stability = no rare large errors), not average MAE. Get any wrong
and random ties or wins. That is exactly what NVIDIA/ALCHEMI exploits.

Caveats (honest): MACE-large stands in for DFT; 6 restarts; stand-in Delta could be
swapped for a real GPU-DFT oracle. The loop structure is production-identical.

## CORRECTION (measured): three defects in the run above; result survives, numbers move

The +35-62% above was produced by code carrying three bugs (element-table loss, a
rotation-dependent acquisition score, best-epoch selection on 2-3 validation configs).
All fixed; re-running the same pool over 6 restarts gave +26-59%, with AL winning 6/6
paired restarts at every budget (one-sided sign test, p = 0.016 each) and the effect
reaching the median for the first time. Points moved in BOTH directions (-12% at 84
labels, +8% at 72) -- the signature of removing noise, not bias. The old +62% was the
maximum over a five-point sweep and did not reproduce; quoting the best point of a
sweep quotes the noise along with the effect. Details in the next section.

## RESULT (real DFT): the effect is not an artefact of the proxy oracle

Everything above used MACE-OFF *large* as the reference. The open question was whether
the active-learning advantage was partly an artefact of student and oracle sharing an
architecture family and a training distribution -- a same-family teacher is a target
the student is unusually well suited to reproduce, which would inflate the measured
gap. The honest expectation was that the advantage would shrink against real DFT.

It did not. 536 configurations relabelled at **PBE0/def2-SVP** (PySCF), 536/536 SCF
converged, 36 min; the same pool, the same loop, 6 paired restarts, 703 min.

| configs labeled | 36 | 48 | 60 | 72 | 84 |
|:---|---:|---:|---:|---:|---:|
| MACE-large oracle | +26% | +49% | +53% | +59% | +50% |
| **PBE0/def2-SVP**  | **+24%** | **+45%** | **+45%** | **+55%** | **+50%** |
| restarts AL wins (DFT) | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |

Within noise of each other. The correlated-oracle critique was real and worth closing,
and closing it cost 36 minutes of DFT -- but it was not where the problem was.

### The actual find: the oracle saturates, so the filter never fired

Real DFT on the same geometries disagreed with the MACE labels in a structured way.
On the worst configurations DFT reaches 40-80 eV/A where the MACE label reports
10-23. MACE-OFF is trained on SPICE (near-equilibrium MD, 300-500 K); a 0.66 A bond is
40% compressed, far outside anything it has seen, and rather than extrapolate up the
repulsive wall it flattens.

`build_ft_pool.py` filters unphysical configurations at `|F|max >= 40 eV/A`, computed
from that label. The highest label anywhere in the 536-config pool was **27.2 eV/A**:
the filter had **never once fired**. Near-dissociated geometries passed straight
through and concentrated in the hard tail, so half the held-out test set carried a
bond under 0.9 A and 10% carried one under 0.7 A. Against DFT the same filter removes
42 configurations (7.8%), and the test set's shortest bond goes 0.64 -> 0.71 A.

Two consequences. The README's claim that the test regime is "what a real MD run
visits" was wrong -- a real trajectory does not visit 0.66 A bonds. And some of the
old absolute p90 (2000-5000 meV/A, against 1088-1723 now) was the student failing to
reproduce a saturation artefact on geometries that should never have been in the set.
The AL-vs-random comparison was never invalid, since both arms saw identical labels.

The mechanism, checked rather than asserted: **MACE-OFF23 carries no ZBL pair-repulsion
term** (confirmed on both the small and large checkpoints). Nothing in the architecture
forces divergence at short range, so the model is a smooth regressor extrapolating off
its manifold and reverts to a bounded value while the true interaction climbs. The
error is signed -- it under-predicts hardest exactly where forces are largest -- which
is what defeats a threshold test specifically, as opposed to merely adding noise to it.

The generalisable claim is therefore narrower than "a universal MLIP cannot police its
own training data": **check whether the potential has an explicit short-range repulsion
term before filtering geometries with it.** MACE-MP ships a ZBL core and should not
fail this way, which matters because it is the permissive alternative this project
recommends elsewhere. One model family measured here; the argument for why it
generalises is mechanistic, not empirical.

### Also fixed, for the record

Three ordinary defects were found and fixed before these runs: fine-tuned models were
rebuilding the element table from their train file (`--foundation_model_elements`
defaults to False), so a bootstrap draw missing a molecule produced a model that
raises on it rather than degrading; the acquisition score averaged per-component
standard deviations and so was not rotation-invariant (~3.5% under rigid rotation,
silently reordering the queue); and best-epoch selection ran on the 2-3 validation
configurations left by `--valid_fraction=0.1` on a 24-config train file, replaced by a
fixed 40-config validation set shared across the study. Re-running moved individual
points in both directions, which is what removing noise looks like rather than
removing bias.

## Explicitly out of scope on this hardware

Running the CUDA/warp-lang Toolkit-Ops kernels, DGX/Run:ai/K8s orchestration,
Omniverse. Named and understood, not faked. A cloud GPU box (or the job) is where
those get hands-on.
