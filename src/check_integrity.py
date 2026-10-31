"""Verify the invariants the rest of the pipeline assumes.

The report claims the results are reproducible and that the classifier never saw its
test data. Those are claims about invariants, and invariants that are never checked
tend to quietly stop holding. This script checks them.

Every later step joins embeddings, cluster labels and abstracts *by row position*, so
a single misaligned file would corrupt every number downstream without raising an
error anywhere. That is the main thing being guarded here.

Usage:   python check_integrity.py [--phase pilot|main|both]
Exit code is non-zero if any check fails, so this can gate a commit.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from config import (
    EMBEDDING_MODELS, PHASES, SPLIT_FRACTIONS, SPLITS_DIR, YEAR_MIN,
    MIN_ABSTRACT_WORDS, group_of, work_path,
)

PASS, FAIL = "  [ok]  ", "  [FAIL]"


class Checker:
    def __init__(self):
        self.failures = 0

    def check(self, condition, label, detail=""):
        if condition:
            print(f"{PASS} {label}" + (f"  ({detail})" if detail else ""))
        else:
            print(f"{FAIL} {label}" + (f"  ({detail})" if detail else ""))
            self.failures += 1
        return bool(condition)


def check_phase(phase, c):
    print(f"\n[{phase}]")
    sample_path = work_path(phase, "sample.parquet")
    if not c.check(sample_path.exists(), "sample.parquet exists"):
        return

    df = pd.read_parquet(sample_path)
    spec = PHASES[phase]

    # --- the sample itself -------------------------------------------------------
    c.check(len(df) == spec["total"], "sample size matches config",
            f"{len(df):,} rows")
    c.check(df["id"].is_unique, "no duplicate papers")
    c.check(df["abstract"].notna().all(), "no missing abstracts")
    c.check((df["year"] >= YEAR_MIN).all(), f"all papers from {YEAR_MIN} onward",
            f"earliest {int(df['year'].min())}")
    short = (df["abstract"].str.split().str.len() < MIN_ABSTRACT_WORDS).sum()
    c.check(short == 0, f"no abstract under {MIN_ABSTRACT_WORDS} words",
            f"{short} violations")

    # Stratification is the reason every archive is guaranteed to appear.
    c.check(df["primary_archive"].nunique() == 20, "all 20 active archives present",
            f"{df['primary_archive'].nunique()} found")
    c.check(df["primary_group"].nunique() == 8, "all 8 groups present")
    derived = df["categories"].str.split().str[0].map(group_of)
    c.check((derived == df["primary_group"]).all(),
            "primary_group agrees with the categories column")

    # --- the split ---------------------------------------------------------------
    counts = df["split"].value_counts()
    c.check(set(counts.index) == {"train", "val", "test"}, "three splits present")
    for name, frac in zip(("train", "val", "test"), SPLIT_FRACTIONS):
        share = counts.get(name, 0) / len(df)
        c.check(abs(share - frac) < 0.01, f"{name} is ~{frac:.0%} of the sample",
                f"{share:.1%}")

    # The claim that matters: clustering never saw validation or test abstracts.
    leaked = df.loc[df["in_cluster_set"], "split"].ne("train").sum()
    c.check(leaked == 0, "clustering subset is drawn from train only",
            f"{leaked} non-train rows flagged")
    c.check(df["in_cluster_set"].sum() == min(spec["cluster"],
                                              int(counts.get("train", 0))),
            "clustering subset is the configured size",
            f"{int(df['in_cluster_set'].sum()):,}")

    # Stratification must survive the split, or rare archives vanish from val/test.
    missing = [s for s in ("train", "val", "test")
               if df[df.split == s]["primary_archive"].nunique() < 20]
    c.check(not missing, "every split contains all 20 archives",
            f"incomplete: {missing}" if missing else "")

    # --- committed ids match the sample -------------------------------------------
    ids_file = SPLITS_DIR / f"{phase}_ids.csv.gz"
    if c.check(ids_file.exists(), f"{ids_file.name} is committed"):
        # dtype=str is required: arXiv ids look numeric, so pandas parses them as
        # float64 by default, which would drop a leading zero on an old-style id.
        ids = pd.read_csv(ids_file, dtype={"id": str})
        c.check(len(ids) == len(df), "committed id count matches the sample")
        c.check(list(ids["id"].astype(str)) == list(df["id"].astype(str)),
                "committed ids match the sample, in order")
        c.check((ids["split"].to_numpy() == df["split"].to_numpy()).all(),
                "committed split assignment matches")

    # --- embeddings ---------------------------------------------------------------
    for key, spec_model in EMBEDDING_MODELS.items():
        path = work_path(phase, f"embeddings_{key}.npy")
        if not path.exists():
            print(f"  [skip] embeddings_{key}.npy not built")
            continue
        vectors = np.load(path, mmap_mode="r")
        c.check(len(vectors) == len(df), f"{key}: one row per abstract",
                f"{len(vectors):,} vs {len(df):,}")
        c.check(vectors.shape[1] == spec_model["dim"],
                f"{key}: {spec_model['dim']} dimensions")
        # KMeans in this space assumes Euclidean == cosine, which needs unit norm.
        norms = np.linalg.norm(np.asarray(vectors[:2000]), axis=1)
        c.check(np.allclose(norms, 1.0, atol=1e-3), f"{key}: vectors are L2-normalised",
                f"norm range {norms.min():.4f}-{norms.max():.4f}")

    # --- cluster labels -----------------------------------------------------------
    for path in sorted(work_path(phase, ".").parent.glob(f"{phase}/labels_*.npy")):
        labels = np.load(path)
        name = path.name
        c.check(len(labels) == len(df), f"{name}: one label per abstract",
                f"{len(labels):,}")
        n_clusters = len({int(v) for v in labels if v >= 0})
        noise = float((labels == -1).mean())
        c.check(n_clusters >= 2, f"{name}: at least two categories",
                f"{n_clusters} clusters, {noise:.1%} noise")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main", "both"], default="both")
    args = ap.parse_args()

    c = Checker()
    for phase in (["pilot", "main"] if args.phase == "both" else [args.phase]):
        check_phase(phase, c)

    print()
    if c.failures:
        print(f"{c.failures} check(s) FAILED")
        sys.exit(1)
    print("All integrity checks passed.")


if __name__ == "__main__":
    main()
