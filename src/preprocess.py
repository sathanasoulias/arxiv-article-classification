"""Build the fixed samples and splits that every later step reuses.

One streaming pass over the 5.65 GB snapshot, keeping papers from YEAR_MIN onwards
with a usable abstract, then drawing a stratified proportional sample per phase.

Design notes
------------
* Sampling is stratified on the primary archive, so every category is guaranteed to
  appear rather than being left to chance. Proportions are preserved: rebalancing here
  would make the discovered number of categories an artefact of our sampling.
* The split is computed BEFORE the clustering subset is drawn, and the clustering
  subset comes only from the training portion. The clustering therefore never sees
  validation or test abstracts, which keeps the downstream classifier honest.
* Text cleaning is deliberately minimal (whitespace only). LaTeX is left intact
  because stripping it would remove real signal from mathematics abstracts, and the
  sentence-transformer tokenisers handle it.

Input:   the arXiv snapshot, plus reports/year_mix.json for the archive distribution
Outputs: WORK_DIR/<phase>/sample.parquet   (abstracts; large, regenerable)
         data/splits/<phase>_ids.csv.gz    (ids + split + cluster flag; committed)
         reports/preprocess_summary.json
Usage:   python preprocess.py [--phase pilot|main|both] [--force]
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATASET_VERSION, DATA_FILE, MIN_ABSTRACT_WORDS, PHASES, REPORTS, SEED,
    SPLIT_FRACTIONS, SPLITS_DIR, YEAR_MIN, archive_of, group_of, work_path,
)
from eda_scan import submission_year

WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def allocate(proportions, total, floor):
    """Split `total` across archives proportionally, guaranteeing at least `floor` each."""
    archives = list(proportions)
    base = {a: min(floor, total // len(archives)) for a in archives}
    remaining = total - sum(base.values())

    alloc = dict(base)
    for a in archives:
        alloc[a] += int(round(remaining * proportions[a]))

    # Correct rounding drift so the total is exact.
    drift = total - sum(alloc.values())
    for a in sorted(archives, key=lambda x: -proportions[x]):
        if drift == 0:
            break
        step = 1 if drift > 0 else -1
        if alloc[a] + step >= base[a]:
            alloc[a] += step
            drift -= step
    return alloc


def archive_proportions():
    """2016+ archive distribution, measured in eda_year_mix.py."""
    mix = json.loads((REPORTS / "year_mix.json").read_text(encoding="utf-8"))
    agg = Counter()
    for year, counts in mix["by_year_archive"].items():
        if int(year) >= YEAR_MIN:
            agg.update(counts)
    total = sum(agg.values())
    return {a: c / total for a, c in agg.items()}, agg


def draw_sample(targets):
    """One pass over the snapshot, reservoir-sampling `targets[archive]` rows each."""
    reservoirs = defaultdict(list)
    seen = Counter()
    rng = random.Random(SEED)
    scanned = kept_eligible = 0

    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned % 500_000 == 0:
                print(f"  scanned {scanned:,}...", flush=True)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            cats = (rec.get("categories") or "").split()
            if not cats:
                continue
            year = submission_year(rec)
            if year is None or year < YEAR_MIN:
                continue
            abstract = clean(rec.get("abstract") or "")
            if len(abstract.split()) < MIN_ABSTRACT_WORDS:
                continue

            archive = archive_of(cats[0])
            target = targets.get(archive, 0)
            if target == 0:
                continue

            kept_eligible += 1
            seen[archive] += 1
            row = {
                "id": rec["id"],
                "title": clean(rec.get("title") or ""),
                "abstract": abstract,
                "categories": " ".join(cats),
                "primary_archive": archive,
                "primary_group": group_of(cats[0]),
                "n_labels": len(cats),
                "year": year,
            }

            bucket = reservoirs[archive]
            if len(bucket) < target:
                bucket.append(row)
            else:
                j = rng.randrange(seen[archive])
                if j < target:
                    bucket[j] = row

    rows = [r for bucket in reservoirs.values() for r in bucket]
    return pd.DataFrame(rows), scanned, kept_eligible


def add_splits(df, cluster_n):
    """Stratified 70/15/15 split, then a clustering subset drawn from train only."""
    train_frac, val_frac, _ = SPLIT_FRACTIONS
    strata = df["primary_archive"]

    train_idx, holdout_idx = train_test_split(
        df.index, train_size=train_frac, stratify=strata, random_state=SEED,
    )
    val_share = val_frac / (1 - train_frac)
    val_idx, test_idx = train_test_split(
        holdout_idx, train_size=val_share,
        stratify=strata.loc[holdout_idx], random_state=SEED,
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Clustering never sees val/test.
    df["in_cluster_set"] = False
    cluster_n = min(cluster_n, len(train_idx))
    if cluster_n == len(train_idx):
        df.loc[train_idx, "in_cluster_set"] = True
    else:
        cluster_idx, _ = train_test_split(
            train_idx, train_size=cluster_n,
            stratify=strata.loc[train_idx], random_state=SEED,
        )
        df.loc[cluster_idx, "in_cluster_set"] = True
    return df


def derive_pilot(main_df, spec, proportions):
    """Subsample the pilot from the main sample, so it is a true preview of it.

    Split assignment is inherited, so a paper never moves between train and test
    across phases. The clustering subset is recomputed from the pilot's own training
    rows.
    """
    targets = allocate(proportions, spec["total"], spec["floor"])
    rng = random.Random(SEED)

    picked = []
    for archive, group in main_df.groupby("primary_archive"):
        n = min(targets.get(archive, 0), len(group))
        if n:
            picked.extend(rng.sample(list(group.index), n))

    df = main_df.loc[picked].copy().reset_index(drop=True)

    df["in_cluster_set"] = False
    train_idx = df.index[df["split"] == "train"]
    n = min(spec["cluster"], len(train_idx))
    if n == len(train_idx):
        df.loc[train_idx, "in_cluster_set"] = True
    else:
        cluster_idx, _ = train_test_split(
            train_idx, train_size=n,
            stratify=df.loc[train_idx, "primary_archive"], random_state=SEED,
        )
        df.loc[cluster_idx, "in_cluster_set"] = True
    return df


def describe(df, scanned=None, eligible=None):
    counts = df["primary_archive"].value_counts()
    return {
        "n_papers": len(df),
        "scanned": scanned,
        "eligible_after_filters": eligible,
        "archives": int(df["primary_archive"].nunique()),
        "groups": int(df["primary_group"].nunique()),
        "splits": {k: int(v) for k, v in df["split"].value_counts().items()},
        "cluster_set": int(df["in_cluster_set"].sum()),
        "largest_archive": f"{counts.idxmax()} ({int(counts.max()):,})",
        "smallest_archive": f"{counts.idxmin()} ({int(counts.min()):,})",
        "multi_label_pct": round(100 * (df["n_labels"] > 1).mean(), 1),
        "abstract_words_median": int(df["abstract"].str.split().str.len().median()),
    }


def save_phase(name, df):
    parquet = work_path(name, "sample.parquet")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, index=False)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    ids_file = SPLITS_DIR / f"{name}_ids.csv.gz"
    df[["id", "primary_archive", "primary_group", "split", "in_cluster_set"]].to_csv(
        ids_file, index=False, compression="gzip"
    )

    print(f"[{name}] {len(df):,} papers -> {parquet.stat().st_size / 1e6:.1f} MB")
    print(f"[{name}] splits: " + ", ".join(
        f"{k}={v:,}" for k, v in df["split"].value_counts().items())
          + f", cluster set={int(df['in_cluster_set'].sum()):,}")
    print(f"[{name}] ids committed to {ids_file.relative_to(SPLITS_DIR.parent.parent)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    main_parquet = work_path("main", "sample.parquet")
    if main_parquet.exists() and not args.force:
        print("Already built. Use --force to rebuild.")
        return

    proportions, counts = archive_proportions()
    print(f"{YEAR_MIN}+ distribution: {len(proportions)} archives, "
          f"{sum(counts.values()):,} papers available\n")

    # One pass for the main sample; the pilot is a subsample of it.
    spec = PHASES["main"]
    targets = allocate(proportions, spec["total"], spec["floor"])
    print(f"Drawing {spec['total']:,} papers across {len(targets)} archives...")
    main_df, scanned, eligible = draw_sample(targets)
    main_df = add_splits(main_df, spec["cluster"])
    print(f"scanned {scanned:,} records, {eligible:,} passed filters\n")

    pilot_df = derive_pilot(main_df, PHASES["pilot"], proportions)

    save_phase("main", main_df)
    save_phase("pilot", pilot_df)

    summary = {
        "dataset_version": DATASET_VERSION,
        "seed": SEED,
        "year_min": YEAR_MIN,
        "min_abstract_words": MIN_ABSTRACT_WORDS,
        "split_fractions": list(SPLIT_FRACTIONS),
        "main": describe(main_df, scanned, eligible),
        "pilot": describe(pilot_df),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {REPORTS / 'preprocess_summary.json'}")


if __name__ == "__main__":
    main()
