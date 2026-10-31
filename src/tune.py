"""Hyperparameter search and encoder comparison for the classifier.

Each configuration runs as a separate subprocess invocation of train_classifier.py.
That is slower than looping in-process, but it means every row of the results table
corresponds to a command that can be re-run on its own, and no state leaks between
runs.

Search design
-------------
Two stages, rather than one large grid, because the grid is multiplicative and most of
it is wasted:

  stage 1 (lr)       learning rate over 1e-5 .. 5e-5 at a fixed batch size. Learning
                     rate dominates fine-tuning outcomes, so it is resolved first.
  stage 2 (batch)    batch size at the winning learning rate.
  stage 3 (encoder)  bert-base vs DistilBERT vs SciBERT at the winning settings.
                     SciBERT is included because the corpus is scientific text, even
                     though Arcan (2026) found its sentence embeddings were the worst
                     of four for clustering — embedding quality and fine-tuning
                     quality are different properties and the comparison is cheap.
  stage 4 (seeds)    the winning configuration re-run on three seeds, so the headline
                     number carries a spread rather than implying a precision it does
                     not have.

Selection uses validation macro-F1 only. Test is never consulted during the search;
train_classifier.py scores it once per run and those numbers are reported but not
compared.

Outputs: reports/tuning_<phase>_<labels>.csv   (one row per configuration)
         reports/tuning_<phase>_<labels>.json  (winner + the full search record)
Usage:   python tune.py --phase pilot --labels hdbscan_mcs100
         python tune.py --phase main --labels kmeans_k7 --stages lr,batch
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from config import REPORTS

HERE = Path(__file__).resolve().parent

LR_GRID = [1e-5, 2e-5, 3e-5, 5e-5]
BATCH_GRID = [8, 16, 32]
ENCODER_GRID = ["bert-base", "distilbert", "scibert"]
SEED_GRID = [42, 1337, 2024]


def run_one(phase, labels, embed_model, encoder, lr, batch, epochs, seed,
            max_train, dry_run):
    """One training run. Returns its metrics dict, or None if it failed."""
    tag = (f"tune_{phase}_{encoder}_{labels}_lr{lr:g}_bs{batch}"
           f"_ep{epochs:g}_seed{seed}")
    out = REPORTS / f"classifier_{tag}.json"

    cmd = [
        sys.executable, str(HERE / "train_classifier.py"),
        "--phase", phase, "--labels", labels, "--embed-model", embed_model,
        "--encoder", encoder, "--lr", str(lr), "--batch-size", str(batch),
        "--epochs", str(epochs), "--seed", str(seed), "--tag", tag,
    ]
    if max_train:
        cmd += ["--max-train", str(max_train)]

    if dry_run:
        print("  " + " ".join(cmd[1:]))
        return None

    if out.exists():
        print(f"  [cached] {tag}")
        return json.loads(out.read_text(encoding="utf-8"))

    print(f"  running {tag} ...", flush=True)
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        print(f"  FAILED ({proc.returncode}) after {time.time() - start:.0f}s")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
        return None

    res = json.loads(out.read_text(encoding="utf-8"))
    print(f"  {tag}: val macro-F1={res['results']['val']['macro_f1']:.4f}  "
          f"test macro-F1={res['results']['test']['macro_f1']:.4f}  "
          f"({res['train_minutes']:.1f} min)")
    return res


def row_of(res):
    """Flatten one run into a results-table row."""
    hp = res["hyperparameters"]
    return {
        "tag": res["tag"],
        "encoder": res["encoder"],
        "learning_rate": hp["learning_rate"],
        "batch_size": hp["batch_size"],
        "epochs": hp["epochs"],
        "seed": hp["seed"],
        "weighted_loss": hp["weighted_loss"],
        "val_macro_f1": res["results"]["val"]["macro_f1"],
        "val_accuracy": res["results"]["val"]["accuracy"],
        "test_macro_f1": res["results"]["test"]["macro_f1"],
        "test_accuracy": res["results"]["test"]["accuracy"],
        "train_minutes": res["train_minutes"],
    }


def best_of(rows, stage):
    """Highest validation macro-F1 among this stage's runs."""
    staged = [r for r in rows if r["stage"] == stage]
    return max(staged, key=lambda r: r["val_macro_f1"]) if staged else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--embed-model", choices=["minilm", "mpnet"], default="minilm")
    ap.add_argument("--labels", default="hdbscan_mcs100")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--stages", default="lr,batch,encoder,seeds",
                    help="comma-separated subset of lr,batch,encoder,seeds")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and the total run count, train nothing")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    common = dict(phase=args.phase, labels=args.labels,
                  embed_model=args.embed_model, epochs=args.epochs,
                  max_train=args.max_train, dry_run=args.dry_run)

    rows = []
    # Defaults carried forward between stages; each stage overwrites one of them.
    encoder, batch, lr, seed = "bert-base", 16, 2e-5, 42

    if "lr" in stages:
        print(f"\nStage 1 — learning rate (encoder={encoder}, batch={batch})")
        for value in LR_GRID:
            res = run_one(encoder=encoder, lr=value, batch=batch, seed=seed, **common)
            if res:
                rows.append({**row_of(res), "stage": "lr"})
        winner = best_of(rows, "lr")
        if winner:
            lr = winner["learning_rate"]
            print(f"  -> lr={lr:g} (val macro-F1 {winner['val_macro_f1']:.4f})")

    if "batch" in stages:
        print(f"\nStage 2 — batch size (lr={lr:g})")
        for value in BATCH_GRID:
            if value == batch and any(r["stage"] == "lr"
                                      and r["batch_size"] == value
                                      and r["learning_rate"] == lr for r in rows):
                continue   # already measured in stage 1
            res = run_one(encoder=encoder, lr=lr, batch=value, seed=seed, **common)
            if res:
                rows.append({**row_of(res), "stage": "batch"})
        pool = [r for r in rows if r["stage"] in ("lr", "batch")
                and r["learning_rate"] == lr]
        if pool:
            batch = max(pool, key=lambda r: r["val_macro_f1"])["batch_size"]
            print(f"  -> batch={batch}")

    if "encoder" in stages:
        print(f"\nStage 3 — encoder (lr={lr:g}, batch={batch})")
        for value in ENCODER_GRID:
            res = run_one(encoder=value, lr=lr, batch=batch, seed=seed, **common)
            if res:
                rows.append({**row_of(res), "stage": "encoder"})
        winner = best_of(rows, "encoder")
        if winner:
            encoder = {v: k for k, v in {
                "bert-base": "bert-base-uncased",
                "distilbert": "distilbert-base-uncased",
                "scibert": "allenai/scibert_scivocab_uncased",
            }.items()}.get(winner["encoder"], encoder)
            print(f"  -> {encoder} (val macro-F1 {winner['val_macro_f1']:.4f})")

    if "seeds" in stages:
        print(f"\nStage 4 — seed spread ({encoder}, lr={lr:g}, batch={batch})")
        for value in SEED_GRID:
            res = run_one(encoder=encoder, lr=lr, batch=batch, seed=value, **common)
            if res:
                rows.append({**row_of(res), "stage": "seeds"})

    if args.dry_run:
        n = (len(LR_GRID) * ("lr" in stages) + len(BATCH_GRID) * ("batch" in stages)
             + len(ENCODER_GRID) * ("encoder" in stages)
             + len(SEED_GRID) * ("seeds" in stages))
        print(f"\n{n} training runs would be launched.")
        return

    if not rows:
        print("\nNo runs completed.")
        return

    df = pd.DataFrame(rows)
    stem = f"tuning_{args.phase}_{args.labels}"
    df.to_csv(REPORTS / f"{stem}.csv", index=False)

    overall = df.loc[df["val_macro_f1"].idxmax()]
    seeds = df[df["stage"] == "seeds"]

    print("\nAll runs, ranked by validation macro-F1:")
    print(df.sort_values("val_macro_f1", ascending=False)[
        ["stage", "encoder", "learning_rate", "batch_size", "seed",
         "val_macro_f1", "test_macro_f1", "train_minutes"]
    ].to_string(index=False))

    summary = {
        "phase": args.phase,
        "labels": args.labels,
        "stages": stages,
        "n_runs": len(df),
        "selection_metric": "validation macro-F1; test is never used for selection",
        "best": {
            "tag": overall["tag"],
            "encoder": overall["encoder"],
            "learning_rate": float(overall["learning_rate"]),
            "batch_size": int(overall["batch_size"]),
            "epochs": float(overall["epochs"]),
            "seed": int(overall["seed"]),
            "val_macro_f1": float(overall["val_macro_f1"]),
            "test_macro_f1": float(overall["test_macro_f1"]),
        },
        "seed_spread": ({
            "seeds": [int(s) for s in seeds["seed"]],
            "test_macro_f1_mean": round(float(seeds["test_macro_f1"].mean()), 4),
            "test_macro_f1_sd": round(float(seeds["test_macro_f1"].std()), 4),
        } if len(seeds) > 1 else None),
        "runs": rows,
    }
    (REPORTS / f"{stem}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBest: {overall['encoder']} lr={overall['learning_rate']:g} "
          f"batch={int(overall['batch_size'])} -> "
          f"val macro-F1 {overall['val_macro_f1']:.4f}, "
          f"test macro-F1 {overall['test_macro_f1']:.4f}")
    if summary["seed_spread"]:
        s = summary["seed_spread"]
        print(f"Seed spread: test macro-F1 {s['test_macro_f1_mean']:.4f} "
              f"+/- {s['test_macro_f1_sd']:.4f} over {len(s['seeds'])} seeds")
    print(f"Wrote {stem}.csv and {stem}.json")


if __name__ == "__main__":
    main()
