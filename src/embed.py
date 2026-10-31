"""Encode abstracts into sentence embeddings and cache them to disk.

The encoder is FROZEN — nothing is trained here. Every paper reviewed for this project
uses off-the-shelf embeddings for scientific-document clustering; fine-tuning an
encoder is a separate research problem and the evidence suggests it would not help.

Only the abstract is encoded, with no title. The brief specifies abstracts, and Arcan
(2026) found abstract-only input beat every richer variant they tried.

Vectors are L2-normalised, so Euclidean distance and cosine similarity become
equivalent — which is what KMeans assumes and what the clustering step relies on.

Input:   WORK_DIR/<phase>/sample.parquet
Outputs: WORK_DIR/<phase>/embeddings_<model>.npy   (float32, L2-normalised)
         WORK_DIR/<phase>/embeddings_<model>.json  (provenance)
Usage:   python embed.py [--phase pilot|main|both] [--model minilm|mpnet|both] [--force]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from config import EMBED_BATCH_SIZE, EMBEDDING_MODELS, SEED, work_path


def embed_phase(phase, model_key, force):
    out_npy = work_path(phase, f"embeddings_{model_key}.npy")
    out_meta = work_path(phase, f"embeddings_{model_key}.json")

    if out_npy.exists() and not force:
        arr = np.load(out_npy, mmap_mode="r")
        print(f"[{phase}/{model_key}] cached: {arr.shape} — skipping (--force to redo)")
        return

    sample = work_path(phase, "sample.parquet")
    if not sample.exists():
        raise FileNotFoundError(f"{sample} missing — run preprocess.py first")

    df = pd.read_parquet(sample, columns=["id", "abstract"])
    texts = df["abstract"].tolist()

    spec = EMBEDDING_MODELS[model_key]
    print(f"\n[{phase}/{model_key}] encoding {len(texts):,} abstracts "
          f"with {spec['name']}")

    # Imported here so --help works without the heavy dependency loaded.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["name"])
    print(f"[{phase}/{model_key}] max_seq_length={model.max_seq_length}, "
          f"dim={model.get_sentence_embedding_dimension()}")

    start = time.time()
    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.time() - start

    np.save(out_npy, vectors)
    out_meta.write_text(json.dumps({
        "phase": phase,
        "model_key": model_key,
        "model_name": spec["name"],
        "n_rows": len(texts),
        "dim": int(vectors.shape[1]),
        "max_seq_length": int(model.max_seq_length),
        "normalised": True,
        "input": "abstract only",
        "batch_size": EMBED_BATCH_SIZE,
        "seed": SEED,
        "seconds": round(elapsed, 1),
        "rows_per_second": round(len(texts) / elapsed, 1),
    }, indent=2), encoding="utf-8")

    print(f"[{phase}/{model_key}] {vectors.shape} -> {out_npy.name} "
          f"({out_npy.stat().st_size / 1e6:.0f} MB) in {elapsed / 60:.1f} min "
          f"({len(texts) / elapsed:.0f} rows/s)")

    # Row order must match sample.parquet, since later steps join on position.
    assert len(vectors) == len(df), "embedding count does not match sample"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main", "both"], default="both")
    ap.add_argument("--model", choices=["minilm", "mpnet", "both"], default="minilm")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    phases = ["pilot", "main"] if args.phase == "both" else [args.phase]
    models = ["minilm", "mpnet"] if args.model == "both" else [args.model]

    for model_key in models:
        for phase in phases:
            embed_phase(phase, model_key, args.force)


if __name__ == "__main__":
    main()
