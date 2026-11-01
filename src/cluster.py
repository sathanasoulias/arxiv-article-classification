"""Sweep clustering configurations and score each one.

This is the step that answers "how many categories are there".

Method note — the key difference from Arcan (2026)
--------------------------------------------------
Arcan selected the number of clusters by maximising 0.5*ARI + 0.5*NMI computed
against arXiv's true labels. That is circular when the brief states the number of
categories is unknown: it uses the answer to find the answer, and the paper itself
concedes it "may inflate reported ARI/NMI".

Here, ARI and AMI are computed and reported but are NEVER used to choose k. Selection
uses internal measures only (silhouette for KMeans, DBCV for HDBSCAN) plus the
stability analysis in stability.py. The external scores are independent validation,
calculated after the fact.

AMI is reported alongside ARI because raw NMI rises with the number of clusters and
would unfairly reward splitting the data further.

Both reference levels are reported: the 20 archives and the 8 groups. Neither is
correct alone — 8 groups treat 13 distinct physics fields as one class, while 20
archives treat all of computer science as one class.

Input:   WORK_DIR/<phase>/embeddings_<model>.npy, sample.parquet
Outputs: WORK_DIR/<phase>/umap_<model>_<dims>d.npy      (cached reduction)
         reports/clustering_<phase>_<model>.csv
Usage:   python cluster.py --phase pilot --model minilm [--force]
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, silhouette_score,
)

from config import REPORTS, SEED, work_path

warnings.filterwarnings("ignore")

K_RANGE = range(3, 21)
UMAP_DIMS = 50            # for clustering; 2D is produced separately for plots
UMAP_NEIGHBOURS = 15
SILHOUETTE_SAMPLE = 10_000   # silhouette is O(n^2); sample instead
DBCV_SAMPLE = 5_000
HDBSCAN_MAX_N = 50_000       # above this HDBSCAN exhausts memory; see run_hdbscan


def reduce_dims(vectors, phase, model, dims, force):
    """UMAP reduction, cached. Needed before density clustering.

    Arcan ran HDBSCAN on raw high-dimensional vectors with no reduction and it
    collapsed to 3-5 clusters with up to 74% noise. Density estimates degrade badly
    in high dimensions, so reducing first is the standard recipe (and what BERTopic
    does).
    """
    out = work_path(phase, f"umap_{model}_{dims}d.npy")
    if out.exists() and not force:
        print(f"  cached UMAP {dims}d")
        return np.load(out)

    import umap
    print(f"  UMAP -> {dims}d (n_neighbors={UMAP_NEIGHBOURS}, metric=cosine)...",
          flush=True)
    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=UMAP_NEIGHBOURS,
        min_dist=0.0,          # tight clumps help density clustering
        metric="cosine",
        random_state=SEED,     # reproducible, at the cost of single-threaded
    )
    reduced = reducer.fit_transform(vectors).astype(np.float32)
    np.save(out, reduced)
    print(f"  saved {out.name}  {reduced.shape}")
    return reduced


def external_scores(labels, archives, groups, mask=None):
    """ARI/AMI against both reference levels. Validation only, never for selection."""
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return dict.fromkeys(
            ["ari_archive", "ami_archive", "ari_group", "ami_group"], np.nan)
    return {
        "ari_archive": adjusted_rand_score(archives[mask], labels[mask]),
        "ami_archive": adjusted_mutual_info_score(archives[mask], labels[mask]),
        "ari_group": adjusted_rand_score(groups[mask], labels[mask]),
        "ami_group": adjusted_mutual_info_score(groups[mask], labels[mask]),
    }


def sampled_silhouette(X, labels):
    if len(set(labels)) < 2:
        return np.nan
    return silhouette_score(
        X, labels, sample_size=min(SILHOUETTE_SAMPLE, len(X)), random_state=SEED
    )


def cluster_shape(labels, n_total):
    """Size distribution: catches a run where one cluster swallows everything."""
    sizes = pd.Series(labels).value_counts()
    sizes = sizes[sizes.index >= 0]
    return {
        "largest_cluster_pct": round(100 * sizes.max() / n_total, 1),
        "smallest_cluster_n": int(sizes.min()),
    }


def run_kmeans(X, archives, groups):
    rows = []
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
        row = {
            "algorithm": "kmeans",
            "param": f"k={k}",
            "n_clusters": k,
            "noise_pct": 0.0,
            "silhouette": sampled_silhouette(X, labels),
            "dbcv": np.nan,
            **cluster_shape(labels, len(X)),
            **external_scores(labels, archives, groups),
        }
        rows.append(row)
        print(f"  kmeans k={k:2}  sil={row['silhouette']:.4f}  "
              f"ARI(arch)={row['ari_archive']:.4f}  AMI(grp)={row['ami_group']:.4f}  "
              f"largest={row['largest_cluster_pct']:.1f}%", flush=True)
    return rows


def run_hdbscan(X, archives, groups, max_n=HDBSCAN_MAX_N):
    """HDBSCAN sweep, on a subsample when the corpus is large.

    HDBSCAN builds a mutual-reachability structure whose memory grows far faster than
    linearly. On the full 150,000-paper sample it reached 16 GB of paged memory on a
    32 GB machine and made no progress, so the sweep runs on a seeded subsample. KMeans
    has no such problem and still runs on everything.
    """
    from hdbscan.validity import validity_index

    n_full = len(X)
    if len(X) > max_n:
        idx = np.sort(np.random.default_rng(SEED).choice(len(X), max_n, replace=False))
        X, archives, groups = X[idx], archives[idx], groups[idx]
        print(f"  (subsampled to {max_n:,} of {n_full:,} papers — see docstring)")

    rows = []
    # Scaled to the dataset: a cluster must be at least this many papers to count.
    sizes = sorted({max(25, int(len(X) * f)) for f in (0.002, 0.005, 0.01, 0.02, 0.04)})
    for mcs in sizes:
        labels = HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(X)
        noise = labels == -1
        n_clusters = len(set(labels[~noise]))

        dbcv = np.nan
        if n_clusters >= 2:
            idx = np.random.default_rng(SEED).choice(
                np.flatnonzero(~noise), size=min(DBCV_SAMPLE, int((~noise).sum())),
                replace=False)
            try:
                dbcv = validity_index(
                    X[idx].astype(np.float64), labels[idx], metric="euclidean")
            except Exception:
                pass

        row = {
            "algorithm": "hdbscan",
            "param": f"min_cluster_size={mcs}",
            "n_clusters": n_clusters,
            "noise_pct": round(100 * noise.mean(), 1),
            # silhouette assumes convex clusters, so it is invalid here; DBCV replaces it
            "silhouette": np.nan,
            "dbcv": dbcv,
            **(cluster_shape(labels, len(X)) if n_clusters else
               {"largest_cluster_pct": np.nan, "smallest_cluster_n": 0}),
            **external_scores(labels, archives, groups, mask=~noise),
        }
        rows.append(row)
        print(f"  hdbscan mcs={mcs:5}  clusters={n_clusters:3}  "
              f"noise={row['noise_pct']:5.1f}%  dbcv={dbcv:.4f}  "
              f"ARI(arch)={row['ari_archive']:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--model", choices=["minilm", "mpnet"], default="minilm")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    vectors = np.load(work_path(args.phase, f"embeddings_{args.model}.npy"))
    df = pd.read_parquet(
        work_path(args.phase, "sample.parquet"),
        columns=["primary_archive", "primary_group"],
    )
    assert len(vectors) == len(df)
    archives = df["primary_archive"].to_numpy()
    groups = df["primary_group"].to_numpy()

    print(f"\n[{args.phase}/{args.model}] {vectors.shape[0]:,} papers, "
          f"{vectors.shape[1]}d embeddings")
    X = reduce_dims(vectors, args.phase, args.model, UMAP_DIMS, args.force)

    out = REPORTS / f"clustering_{args.phase}_{args.model}.csv"

    def flush(rows):
        """Write after each sweep, so an interrupted run keeps what it finished."""
        result = pd.DataFrame(rows)
        result.insert(0, "phase", args.phase)
        result.insert(1, "model", args.model)
        result.insert(2, "umap_dims", UMAP_DIMS)
        result.to_csv(out, index=False)
        return result

    print(f"\nKMeans sweep k={K_RANGE.start}..{K_RANGE.stop - 1}")
    rows = run_kmeans(X, archives, groups)
    flush(rows)
    print(f"  KMeans results saved to {out.name}")

    print("\nHDBSCAN sweep")
    rows += run_hdbscan(X, archives, groups)
    result = flush(rows)
    print(f"\nWrote {out}")

    best = result[result.algorithm == "kmeans"].nlargest(3, "silhouette")
    print("\nTop 3 by silhouette (selection metric, NOT ARI):")
    print(best[["param", "silhouette", "ari_archive", "ami_group",
                "largest_cluster_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
