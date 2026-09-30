# Project 4 — MLIPs, GPU atomistics, ALCHEMI

**Status: local only. Not on GitHub, not on the resume yet.**

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
"oracle" is pluggable — a cheap analytic reference now, a GPU-DFT call in production.

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

Caveats (honest): MACE-large stands in for DFT; 2 restarts so bands are noisy;
stand-in Delta could be swapped for a real GPU-DFT oracle. The loop structure is
production-identical.

## Explicitly out of scope on this hardware

Running the CUDA/warp-lang Toolkit-Ops kernels, DGX/Run:ai/K8s orchestration,
Omniverse. Named and understood, not faked. A cloud GPU box (or the job) is where
those get hands-on.
