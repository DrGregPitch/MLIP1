#!/usr/bin/env python3
"""Real MLIP active learning: a committee of fine-tuned MACE models selects which
configurations to label (compute the reference for) and fine-tune on.

    python scripts/build_ft_pool.py       # once
    python scripts/run_real_al.py --outdir results

This is the faithful version. The model being fine-tuned is the ACTUAL MACE
foundation model (high capacity -> reducible uncertainty), not a surrogate. A
committee of independently fine-tuned MACEs gives the uncertainty; high force
disagreement flags configurations the committee hasn't learned. We compare
selecting those (active learning) against random selection: held-out force error
vs. number of labeled/fine-tuned configurations.

Robust metrics (median per-config, worst-case p90) on physical configs only.
Traces are checkpointed after every round so a long background run is never lost.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

FM = str(Path.home() / ".cache/mace/MACE-OFF23_small.model")
# invoke the trainer as a module rather than the generated console script: pip bakes an
# absolute interpreter path into .venv/bin/mace_run_train, so a venv that gets copied or
# renamed fails with a FileNotFoundError naming the script, not the missing interpreter.
TRAIN_CMD = [sys.executable, "-m", "mace.cli.run_train"]


def finetune(train_atoms, name, workdir, epochs, seed, e0s="foundation", valid_file=None):
    """Fine-tune MACE-small on train_atoms; return the loadable model path.

    e0s: 'foundation' reuses the foundation model's atomic reference energies, which
    is correct only while the labels sit on the foundation model's own energy scale.
    For labels from a different level of theory (rMD17 ships total PBE/def2-SVP
    energies) pass 'average' so MACE least-squares-fits its own atomic baselines.

    valid_file: a FIXED validation set, shared by every fine-tune in the study.
    Without it MACE carves --valid_fraction=0.1 out of the train file, which is 2-3
    configurations at the small end of the budget sweep; best-epoch selection on
    2-3 configs is close to arbitrary (observed best epochs 50, 28, 66 across
    members of one committee) and injects variance that looks like a result.
    """
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    tf = wd / f"{name}_train.xyz"
    write(tf, train_atoms)
    valid_args = [f"--valid_file={valid_file}"] if valid_file else ["--valid_fraction=0.1"]
    try:
        subprocess.run(TRAIN_CMD + [
            f"--name={name}", f"--foundation_model={FM}",
            f"--train_file={tf}", *valid_args,
            # keep the foundation's full element table. The default (False) rebuilds it
            # from the elements present in THIS train file, so a bootstrap draw that
            # happens to miss a molecule yields a model that cannot be evaluated on it
            # at all -- it raises ValueError: <Z> is not in list, it does not degrade.
            "--foundation_model_elements=True",
            "--energy_key=energy", "--forces_key=forces", f"--E0s={e0s}",
            "--multiheads_finetuning=False",
            f"--max_num_epochs={epochs}", "--batch_size=10", "--lr=0.01",
            "--device=cpu", "--default_dtype=float64", f"--seed={seed}",
            f"--model_dir={wd}", f"--log_dir={wd}", f"--results_dir={wd}",
            f"--checkpoints_dir={wd}",
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # CalledProcessError's message omits captured output; on an overnight
        # run "exit status 1" with no stderr is an unrecoverable landmine
        print(f"[finetune {name}] mace training FAILED; stderr tail:\n"
              f"{(exc.stderr or '')[-2000:]}", flush=True)
        raise
    out = str(wd / f"{name}.model")
    _assert_full_element_table(out)
    return out


def _assert_full_element_table(model_path):
    """Guard: a fine-tuned model must keep the foundation's element table.

    MACE rebuilds that table from the train file unless --foundation_model_elements
    is set, and a model missing an element raises on any config containing it
    rather than degrading. Checking here turns a crash hours into an overnight run
    -- or worse, a silently survivorship-biased set of surviving restarts -- into an
    immediate, legible failure.
    """
    import torch
    zs = {}
    for tag, p in (("foundation", FM), ("finetuned", model_path)):
        m = torch.load(p, map_location="cpu", weights_only=False)
        zs[tag] = m.atomic_numbers.tolist()
    if zs["finetuned"] != zs["foundation"]:
        raise RuntimeError(
            f"fine-tuned element table {zs['finetuned']} != foundation {zs['foundation']}; "
            "--foundation_model_elements=True was not honoured, so this model cannot be "
            "evaluated on configs containing the missing elements"
        )


def load_calc(model_path):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=model_path, device="cpu", default_dtype="float64")


def committee_forces(model_paths, atoms_list):
    """(n_members, n_configs) arrays: per-config force prediction, flattened."""
    preds = []
    for mp in model_paths:
        calc = load_calc(mp)
        row = []
        for a in atoms_list:
            b = a.copy()
            b.calc = calc
            row.append(b.get_forces())
        preds.append(row)
    return preds  # list[member][config] -> (natoms,3)


def per_config_force_rmse(pred_forces, atoms_list):
    """Committee-mean force RMSE per config (meV/A) vs baked-in reference."""
    out = []
    for ci, a in enumerate(atoms_list):
        ref = a.get_forces()
        mean = np.mean([pred_forces[m][ci] for m in range(len(pred_forces))], axis=0)
        out.append(np.sqrt(((mean - ref) ** 2).mean()) * 1000)
    return np.asarray(out)


def committee_disagreement(pred_forces, n_configs):
    """Per-config force disagreement across members (meV/A) -> the AL signal.

    RMS over components, not the mean of per-component standard deviations: the
    mean of |sx|,|sy|,|sz| depends on the coordinate frame, so the same
    configuration scores differently after a rigid rotation and the acquisition
    order is frame-dependent. sqrt(mean(var)) is rotation-invariant because the
    per-atom variance sums as a trace.
    """
    out = []
    for ci in range(n_configs):
        stack = np.stack([pred_forces[m][ci] for m in range(len(pred_forces))])  # (M,nat,3)
        out.append(np.sqrt((stack.std(axis=0) ** 2).mean()) * 1000)
    return np.asarray(out)


def run_campaign(strategy, pool, test, work, args, seed, on_round=None, valid_file=None):
    rng = np.random.default_rng(seed)
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    labeled = idx[: args.seed_size]
    remaining = idx[args.seed_size:]

    n_lab, med, p90 = [], [], []
    rnd = 0
    while True:
        models = []
        for m in range(args.members):
            train = [pool[i] for i in labeled]
            bt = list(rng.integers(0, len(train), len(train)))  # bootstrap for diversity
            mp = finetune([train[i] for i in bt],
                          f"{strategy}_s{seed}_r{rnd}_m{m}",
                          Path(work) / "ft", args.epochs, seed * 100 + m,
                          e0s=args.e0s, valid_file=valid_file)
            models.append(mp)

        tf = committee_forces(models, test)
        e = per_config_force_rmse(tf, test)
        n_lab.append(len(labeled))
        med.append(float(np.median(e)))
        p90.append(float(np.percentile(e, 90)))
        print(f"    [{strategy} seed{seed}] {len(labeled)} labels: "
              f"median {med[-1]:.0f}  p90 {p90[-1]:.0f} meV/A", flush=True)
        if on_round is not None:  # checkpoint the partial trace every round
            on_round({"n_labeled": n_lab, "median_rmse": med, "p90_rmse": p90})

        if len(labeled) >= args.budget or not remaining:
            break
        take = min(args.batch, len(remaining))
        if strategy == "random":
            rng.shuffle(remaining)
            pick = remaining[:take]
        else:  # committee-uncertainty
            pf = committee_forces(models, [pool[i] for i in remaining])
            dis = committee_disagreement(pf, len(remaining))
            order = np.argsort(dis)[::-1][:take]
            pick = [remaining[i] for i in order]
        labeled += pick
        remaining = [i for i in remaining if i not in set(pick)]
        shutil.rmtree(Path(work) / "ft", ignore_errors=True)  # free disk between rounds
        rnd += 1
    return {"n_labeled": n_lab, "median_rmse": med, "p90_rmse": p90}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results", type=Path)
    p.add_argument("--members", default=3, type=int)
    p.add_argument("--epochs", default=80, type=int)
    p.add_argument("--seed-size", default=24, type=int)
    p.add_argument("--batch", default=12, type=int)
    p.add_argument("--budget", default=84, type=int)
    p.add_argument("--restarts", default=2, type=int)
    p.add_argument("--seed-start", default=0, type=int)
    p.add_argument("--work", default="al_work")
    p.add_argument("--pool", default="data_cache/ft_pool.xyz")
    p.add_argument("--test", default="data_cache/ft_heldout.xyz")
    p.add_argument("--e0s", default="foundation", choices=("foundation", "average"))
    p.add_argument("--valid-size", default=40, type=int,
                   help="configs held out of the pool as a FIXED validation set shared by "
                        "every fine-tune; 0 restores MACE's per-train-file --valid_fraction")
    args = p.parse_args()
    args.outdir.mkdir(exist_ok=True)

    if args.members < 2:
        # committee disagreement is the std across members; with one member it is
        # identically zero, so argsort silently returns an arbitrary slice and the
        # "uncertainty" arm quietly stops doing uncertainty selection
        raise SystemExit("--members must be >= 2: committee disagreement is undefined "
                         "for a single model")

    pool = read(args.pool, ":")
    test = read(args.test, ":")

    valid_file = None
    if args.valid_size > 0:
        # one fixed split, seeded independently of the restart seed, so every campaign
        # in the study validates against exactly the same configurations
        vrng = np.random.default_rng(12345)
        vidx = vrng.choice(len(pool), args.valid_size, replace=False)
        vset = set(vidx.tolist())
        valid_atoms = [pool[i] for i in sorted(vset)]
        pool = [pool[i] for i in range(len(pool)) if i not in vset]
        Path(args.work).mkdir(parents=True, exist_ok=True)
        valid_file = str(Path(args.work) / "fixed_valid.xyz")
        write(valid_file, valid_atoms)
        print(f"fixed validation set: {len(valid_atoms)} configs -> {valid_file}")

    # printed after the carve so the pool size shown is the SELECTABLE pool
    print(f"pool {len(pool)} | held-out {len(test)} | committee {args.members} x "
          f"{args.epochs}ep | budget {args.budget} | {args.restarts} restarts | "
          f"E0s={args.e0s}")
    print(f"  pool={args.pool}  test={args.test}")

    trace_file = args.outdir / "real_al_traces.json"
    if trace_file.exists():
        # extending an existing run (e.g. --seed-start 2) must append, not clobber
        results = json.loads(trace_file.read_text())
        print(f"appending to existing traces "
              f"({ {k: len(v) for k, v in results.items()} })", flush=True)
    else:
        results = {"random": [], "uncertainty": []}
    t0 = time.time()
    for r in range(args.seed_start, args.seed_start + args.restarts):
        for strat in ("random", "uncertainty"):
            print(f"\n=== restart {r} / {strat} ({time.time()-t0:.0f}s) ===", flush=True)
            partial = args.outdir / f"partial_{strat}_seed{r}.json"
            res = run_campaign(
                strat, pool, test, args.work, args, seed=r,
                on_round=lambda tr, p=partial: p.write_text(json.dumps(tr)),
                valid_file=valid_file,
            )
            results[strat].append(res)
            partial.unlink(missing_ok=True)  # promoted into the main trace file
            # checkpoint after every campaign
            trace_file.write_text(json.dumps(results, indent=2))
    print(f"\nDONE in {(time.time()-t0)/60:.0f} min. traces -> "
          f"{args.outdir/'real_al_traces.json'}")


if __name__ == "__main__":
    main()
