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

Caveats (honest): MACE-large stands in for DFT; 6 restarts; stand-in Delta could be
swapped for a real GPU-DFT oracle. The loop structure is production-identical.

## CORRECTION (measured): three defects in the run above; result survives, numbers move

The +35-62% above was produced by code carrying three bugs. All three are fixed and
the experiment was re-run from scratch on the same pool, 6 restarts, 602 min. **The
conclusion held and the evidence got stronger, but the headline range came down to
+26-59% and the peak moved from 84 labels to 72.**

| configs labeled | 36 | 48 | 60 | 72 | 84 |
|:---|---:|---:|---:|---:|---:|
| published | +35% | +47% | +55% | +52% | **+62%** |
| corrected | +26% | +49% | +53% | **+59%** | +50% |
| restarts AL wins | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |

Points moved in BOTH directions (-12% at 84, +8% at 72), which is the signature of
removing noise rather than removing a bias. The old +62% was the maximum over a
five-point sweep and did not reproduce -- a reminder that quoting the best point of a
sweep quotes the noise along with the effect.

**The three defects:**

1. **Fine-tuned models silently lost the foundation's element table.** MACE's
   `--foundation_model_elements` defaults to False, so the table is rebuilt from
   whatever is in the train file: a bootstrap draw that missed acetonitrile produced
   a model with no nitrogen, which *raises* on any N-bearing config rather than
   degrading. Latent here because acetonitrile is a third of the pool, but fatal for
   any pool with uneven composition. Now asserted after every fine-tune.
2. **The acquisition score was not rotation-invariant.** `std(axis=0).mean()` is the
   mean of per-*component* standard deviations, and the mean of |sx|,|sy|,|sz| is
   frame-dependent; the same configuration scored ~3.5% differently under rigid
   rotation, silently reordering the acquisition queue. `sqrt(mean(var))` is
   invariant. An exact invariance violation inside a project about equivariant ML.
3. **Best-epoch selection ran on 2-3 validation configs.** `--valid_fraction=0.1` on
   a 24-config train file leaves two or three; observed best epochs across one
   committee were 50, 28 and 66 -- effectively arbitrary. Replaced by a fixed
   40-config validation set carved once and shared by every fine-tune in the study.
   This is also why the corrected run took 602 min against the original 138: forty
   validation configs evaluated every epoch is not free.

**What improved beyond the numbers.** Both arms share the seed set, bootstrap draws
and training seeds, so the comparison is paired and the 24-label point is identical
by construction. AL wins 6/6 paired restarts at every budget (one-sided sign test,
p = 0.016 at each point) -- a far stronger claim than a mean with a standard-error
band over two restarts. The effect also reaches the median for the first time (+17%,
+21%, +21% at the top three budgets), where it previously lived almost entirely in
the tail.

**Lesson for the robust-number list above:** a result can be real and its number
still wrong. Two of these three bugs changed which configurations got selected, and
none announced itself -- the loop ran, the traces looked plausible, and the
conclusion was correct the whole time. Re-running after a fix is not housekeeping; it
is the only way to learn which part of the number was the effect.

## Explicitly out of scope on this hardware

Running the CUDA/warp-lang Toolkit-Ops kernels, DGX/Run:ai/K8s orchestration,
Omniverse. Named and understood, not faked. A cloud GPU box (or the job) is where
those get hands-on.
