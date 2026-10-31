"""Choose the number of categories by how reproducible each grouping is.

Why stability is the primary criterion
--------------------------------------
The brief says the number of categories is unknown, so the number must be chosen
without consulting arXiv's labels. That rules out Arcan (2026), who maximised
0.5*ARI + 0.5*NMI against the true labels and conceded it "may inflate reported
ARI/NMI". It also rules out silhouette alone: in our pilot, silhouette peaked at k=3
on a solution where one cluster held 44% of all papers.

Stability asks a question that needs no labels (Lange, Roth, Braun & Buhmann,
"Stability-Based Validation of Clustering Solutions", Neural Computation 2004,
16(6):1299-1323; the subsample-pair estimator is from Ben-Hur, Elisseeff & Guyon,
PSB 2002). If the data genuinely contains k groups, clustering two different samples
of it should recover the same groups. If k is wrong, the algorithm must invent
boundaries, and invented boundaries land somewhere different each time.

Which space k is decided in — and why it is not UMAP
----------------------------------------------------
cluster.py reduces with UMAP before clustering, because density estimates degrade in
high dimensions and HDBSCAN needs the reduction. But `min_dist=0.0` deliberately
manufactures tightly separated clumps, and clustering inside that space makes every
internal criterion saturate: measured there, stability is 1.000 at k=3 and declines
monotonically, so it would simply reproduce silhouette's degenerate answer. What is
being measured in that case is UMAP's own determinism, not structure in the data.

So k is decided in the original L2-normalised embedding space, which is where the
abstracts actually live and which no stochastic structure-amplifying step has touched.
`--space umap` reproduces the saturated version, because the contrast between the two
is itself a reportable result.

The null baseline
-----------------
ARI is chance-corrected, so random *labellings* score 0. That is not the relevant
null. The question is whether the data has more reproducible structure than
structureless data of the same shape would, since KMeans will happily carve a
featureless blob the same way twice. For the embedding space the null is points drawn
uniformly on the unit sphere of the same dimension — genuinely structureless, and the
right shape because the real vectors are L2-normalised. (An earlier attempt at a null
by shuffling each coordinate independently does NOT work: it preserves each
coordinate's multi-modality, so the product distribution stays clusterable and scores
0.999.)

Input:   WORK_DIR/<phase>/embeddings_<model>.npy   (and umap_*.npy for --space umap)
Outputs: reports/stability_<phase>_<model>_<space>.csv
         reports/stability_<phase>_<model>_<space>.json
Usage:   python stability.py --phase pilot --model minilm
         python stability.py --phase pilot --model minilm --space umap
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from cluster import K_RANGE, UMAP_DIMS
from config import REPORTS, SEED, work_path

SUBSAMPLE_FRAC = 0.8      # two 80% draws overlap on ~64% of the data
DEFAULT_REPS = 10         # pairs per setting; each pair costs two clusterings
MIN_OVERLAP = 50
STABLE_THRESHOLD = 0.80   # Ben-Hur et al.'s cutoff for "a stable solution"


def subsample_pair(n, rng):
    """Two independent draws of the same size, sorted so searchsorted can align them."""
    size = int(n * SUBSAMPLE_FRAC)
    return (np.sort(rng.choice(n, size, replace=False)),
            np.sort(rng.choice(n, size, replace=False)))


def overlap_ari(idx_a, lab_a, idx_b, lab_b):
    """ARI between two clusterings, restricted to the papers both runs saw.

    Noise points (-1, HDBSCAN only) are dropped: an unassigned paper makes no claim
    about grouping, so counting all noise as one shared cluster would reward a run
    for refusing to decide.
    """
    common = np.intersect1d(idx_a, idx_b)
    if len(common) < MIN_OVERLAP:
        return np.nan

    a = lab_a[np.searchsorted(idx_a, common)]
    b = lab_b[np.searchsorted(idx_b, common)]

    keep = (a >= 0) & (b >= 0)
    if keep.sum() < MIN_OVERLAP or len(set(a[keep])) < 2 or len(set(b[keep])) < 2:
        return np.nan
    return adjusted_rand_score(a[keep], b[keep])


def sphere_null(n, dim, rng):
    """Uniform on the unit sphere: same shape as L2-normalised embeddings, no structure."""
    X = rng.standard_normal((n, dim)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def box_null(X, rng):
    """Uniform over each dimension's observed range (Tibshirani et al. 2001, gap statistic).

    Used for spaces that are not on a sphere. Weaker than sphere_null, because a
    bounding box ignores the manifold the data occupies.
    """
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (lo + (hi - lo) * rng.random(X.shape)).astype(np.float32)


def stability(X, fit, reps, seed):
    """Mean/std ARI between pairs of independently subsampled clusterings.

    `fit` maps a data matrix to labels, so one estimator serves KMeans and HDBSCAN.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(reps):
        idx_a, idx_b = subsample_pair(len(X), rng)
        scores.append(overlap_ari(idx_a, fit(X[idx_a]), idx_b, fit(X[idx_b])))
    scores = np.array(scores, dtype=float)
    valid = scores[~np.isnan(scores)]
    if not len(valid):
        return np.nan, np.nan, 0
    return float(valid.mean()), float(valid.std()), len(valid)


def kmeans_fit(k):
    # n_init matches cluster.py: stability must measure the estimator we deploy.
    return lambda Xs: KMeans(n_clusters=k, random_state=SEED,
                             n_init=10).fit_predict(Xs)


def hdbscan_fit(mcs):
    def fit(Xs):
        # Scaled to the subsample, so the density threshold stays the same fraction
        # of the data in every run.
        scaled = max(5, int(round(mcs * SUBSAMPLE_FRAC)))
        return HDBSCAN(min_cluster_size=scaled, metric="euclidean").fit_predict(Xs)
    return fit


def sweep(X, X_null, configs, reps):
    rows = []
    for algorithm, param, value, fit in configs:
        obs, obs_sd, n_obs = stability(X, fit, reps, SEED)
        null, null_sd, _ = stability(X_null, fit, reps, SEED + 1)
        margin = obs - null if not (np.isnan(obs) or np.isnan(null)) else np.nan
        rows.append({
            "algorithm": algorithm,
            "param": f"{param}={value}",
            param: value,
            "stability": obs,
            "stability_sd": obs_sd,
            "null_stability": null,
            "null_sd": null_sd,
            "margin": margin,
            "valid_pairs": n_obs,
        })
        print(f"  {algorithm:8} {param}={value:<5} stability={obs:.4f} "
              f"(+/-{obs_sd:.4f})  null={null:.4f}  margin={margin:+.4f}", flush=True)
    return rows


def select_k(km):
    """Pick k from the stability curve, and say which rule fired.

    Two readings are reported because they answer different questions:
      peak    — the single most reproducible k. Prone to favouring tiny k, since a
                coarse split is trivially easy to reproduce.
      largest — the most granular k that is still stable (>= STABLE_THRESHOLD). Among
                equally reproducible taxonomies the more informative one is preferred,
                because a 2-way split is perfectly stable and useless.

    If stability never falls below the threshold the curve has not discriminated, and
    that is stated rather than papered over.
    """
    peak = km.loc[km["stability"].idxmax()]
    stable = km[km["stability"] >= STABLE_THRESHOLD]
    largest = km.loc[stable["k"].idxmax()] if len(stable) else peak

    monotone = bool(km["stability"].is_monotonic_decreasing)
    exhausted = bool(len(stable) and stable["k"].max() == km["k"].max())
    return {
        "peak_k": int(peak["k"]),
        "peak_stability": round(float(peak["stability"]), 4),
        "largest_stable_k": int(largest["k"]),
        "largest_stable_stability": round(float(largest["stability"]), 4),
        "threshold": STABLE_THRESHOLD,
        "curve_is_monotonic_decreasing": monotone,
        "threshold_never_crossed": exhausted,
        "caveat": (
            "stability declines monotonically, so the peak is uninformative and the "
            "threshold rule carries the decision" if monotone else None
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--model", choices=["minilm", "mpnet"], default="minilm")
    ap.add_argument("--space", choices=["embedding", "umap", "pca50"],
                    default="embedding",
                    help="embedding is the reported setting; umap shows the saturation")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--max-n", type=int, default=20_000,
                    help="cost is reps*2*settings*2 clusterings, so rows are capped")
    ap.add_argument("--skip-hdbscan", action="store_true")
    args = ap.parse_args()

    if args.space == "umap":
        path = work_path(args.phase, f"umap_{args.model}_{UMAP_DIMS}d.npy")
        hint = "run cluster.py first"
    else:
        path = work_path(args.phase, f"embeddings_{args.model}.npy")
        hint = "run embed.py first"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — {hint}")
    X = np.load(path)

    # Capping does not bias the comparison: every setting is measured on identical rows.
    rng = np.random.default_rng(SEED)
    if len(X) > args.max_n:
        X = X[np.sort(rng.choice(len(X), args.max_n, replace=False))]
    n_used = len(X)

    if args.space == "pca50":
        X = PCA(n_components=50, random_state=SEED).fit_transform(X).astype(np.float32)

    # The sphere null is only correct for L2-normalised vectors.
    on_sphere = args.space == "embedding"
    X_null = (sphere_null(n_used, X.shape[1], rng) if on_sphere
              else box_null(X, rng))

    print(f"\n[{args.phase}/{args.model}] stability in {args.space} space: "
          f"{n_used:,} papers, {X.shape[1]}d, {args.reps} subsample pairs per setting "
          f"({SUBSAMPLE_FRAC:.0%} draws)")
    print(f"  null: {'uniform on unit sphere' if on_sphere else 'uniform in bounding box'}")

    configs = [("kmeans", "k", k, kmeans_fit(k)) for k in K_RANGE]
    if not args.skip_hdbscan:
        # HDBSCAN is only run in UMAP space, since it is the step that needs the
        # reduction; in raw high-dimensional space it collapses, which is the Arcan
        # failure cluster.py was written to avoid.
        if args.space == "umap":
            sizes = sorted({max(25, int(n_used * f))
                            for f in (0.002, 0.005, 0.01, 0.02, 0.04)})
            configs += [("hdbscan", "min_cluster_size", m, hdbscan_fit(m))
                        for m in sizes]
        else:
            print("  (HDBSCAN skipped: it is only meaningful after UMAP reduction)")

    print(f"\nKMeans k={K_RANGE.start}..{K_RANGE.stop - 1}")
    rows = sweep(X, X_null, configs, args.reps)

    result = pd.DataFrame(rows)
    for i, (col, val) in enumerate([("phase", args.phase), ("model", args.model),
                                    ("space", args.space), ("n_papers", n_used),
                                    ("reps", args.reps)]):
        result.insert(i, col, val)
    stem = f"stability_{args.phase}_{args.model}_{args.space}"
    result.to_csv(REPORTS / f"{stem}.csv", index=False)

    km = result[result.algorithm == "kmeans"].dropna(subset=["stability"])
    choice = select_k(km)

    print("\nStability curve (KMeans):")
    print(km[["param", "stability", "stability_sd", "null_stability", "margin"]]
          .to_string(index=False))
    print(f"\n  peak:            k={choice['peak_k']} "
          f"(stability {choice['peak_stability']:.4f})")
    print(f"  largest stable:  k={choice['largest_stable_k']} "
          f"(stability {choice['largest_stable_stability']:.4f}, "
          f"threshold {STABLE_THRESHOLD})")
    if choice["caveat"]:
        print(f"  caveat: {choice['caveat']}")
    if choice["threshold_never_crossed"]:
        print(f"  warning: stability never fell below {STABLE_THRESHOLD} within "
              f"k<={K_RANGE.stop - 1}; the sweep may need to extend further")

    summary = {
        "phase": args.phase,
        "model": args.model,
        "space": args.space,
        "n_papers": n_used,
        "reps": args.reps,
        "subsample_frac": SUBSAMPLE_FRAC,
        "seed": SEED,
        "criterion": (
            "mean ARI between independently subsampled clusterings; no external label "
            "is consulted at any point"
        ),
        "null_model": "uniform on unit sphere" if on_sphere
                      else "uniform in per-dimension bounding box",
        "selection": choice,
        "curve": [
            {"k": int(r.k), "stability": round(r.stability, 4),
             "sd": round(r.stability_sd, 4),
             "null": round(r.null_stability, 4),
             "margin": round(r.margin, 4)}
            for r in km.itertuples()
        ],
    }
    (REPORTS / f"{stem}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {stem}.csv and {stem}.json")


if __name__ == "__main__":
    main()
