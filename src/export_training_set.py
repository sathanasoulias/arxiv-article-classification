"""Bundle one training set into a single file for GPU training elsewhere.

Fine-tuning 13 configurations is not feasible on CPU, so the classifier stages run on
a Colab T4. Rather than reproduce the whole pipeline there — which would mean
re-downloading 5.65 GB from Kaggle and re-running UMAP — this exports exactly what the
classifier needs: the abstract, the split it belongs to, and the discovered category
it was assigned.

Everything upstream (sampling, splitting, embedding, clustering) stays here, where it
is seeded and reproducible. The notebook consumes a fixed artifact and cannot
accidentally re-derive the labels differently.

Noise rows are kept with y = -1 rather than dropped, so the notebook can choose how to
handle them and the file documents how many there were.

Input:   WORK_DIR/<phase>/sample.parquet, labels_<model>_<config>.npy
Outputs: WORK_DIR/export/trainset_<phase>_<model>_<config>.parquet
         WORK_DIR/export/trainset_<phase>_<model>_<config>.json   (manifest)
Usage:   python export_training_set.py --phase main --labels kmeans_k7
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from config import MAX_TOKENS, SEED, work_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--embed-model", choices=["minilm", "mpnet"], default="minilm")
    ap.add_argument("--labels", default="hdbscan_mcs100")
    args = ap.parse_args()

    df = pd.read_parquet(
        work_path(args.phase, "sample.parquet"),
        columns=["id", "abstract", "split", "primary_archive", "primary_group"],
    )
    label_file = work_path(args.phase, f"labels_{args.embed_model}_{args.labels}.npy")
    if not label_file.exists():
        raise FileNotFoundError(f"{label_file} missing — run interpret.py first")

    labels = np.load(label_file)
    if len(labels) != len(df):
        raise ValueError(f"{len(labels)} labels for {len(df)} abstracts")
    df["y"] = labels

    stem = f"trainset_{args.phase}_{args.embed_model}_{args.labels}"
    out = work_path("export", f"{stem}.parquet")
    # The archive columns ride along so the notebook can reproduce the composition
    # tables without needing the full sample.
    df.to_parquet(out, index=False, compression="snappy")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    counts = df["y"].value_counts().sort_index()

    manifest = {
        "phase": args.phase,
        "embed_model": args.embed_model,
        "label_config": args.labels,
        "source_labels": label_file.name,
        "n_abstracts": len(df),
        "n_categories": int((counts.index >= 0).sum()),
        "n_noise": int(counts.get(-1, 0)),
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "splits": {k: int(v) for k, v in df["split"].value_counts().items()},
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "file": out.name,
        "bytes": out.stat().st_size,
        "sha256_16": digest,
        "note": (
            "y = -1 marks abstracts HDBSCAN left unassigned; the notebook decides "
            "whether to drop them or treat them as a class. Splits were fixed in "
            "preprocess.py and must not be recomputed."
        ),
    }
    work_path("export", f"{stem}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {out}")
    print(f"  {len(df):,} abstracts, {manifest['n_categories']} categories, "
          f"{manifest['n_noise']:,} noise")
    print(f"  splits: " + ", ".join(f"{k}={v:,}" for k, v in
                                    manifest["splits"].items()))
    print(f"  {out.stat().st_size / 1e6:.1f} MB, sha256:{digest}")
    print(f"\nUpload this one file to Colab (or Google Drive) and point "
          f"notebooks/colab_train.ipynb at it.")


if __name__ == "__main__":
    main()
