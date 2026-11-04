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


def sweep(X, X_null, configs, reps, on_row=None):
    """Score every configuration. `on_row` is called after each one.

    A full sweep is hours of clustering, so each row is handed back as soon as it
    exists and the caller persists it. An interrupted run then keeps everything it
    finished instead of losing the lot.
    """
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
        if on_row:
            on_row(rows)
    return rows


def select_k(km):
    """Pick k from the stability curve, and say which rule fired.

    Ben-Hur et al. test two things together: that the similarity scores are high *and*
    that they are concentrated. A k whose mean clears the threshold on the strength of
    a few lucky resamples is not a stable solution, so the decision is made on a
    one-standard-deviation lower bound rather than on the mean. In our pilot the mean
    alone selects k=11 (mean 0.849, sd 0.067) over k=8 (mean 0.942, sd 0.020), which
    inverts the evidence; the lower bound does not.

    Three readings are reported, because they answer different questions:
      peak      — the single most reproducible k. Biased toward small k, since a
                  coarse split is trivially easy to reproduce.
      selected  — the most granular k that is stable on the lower bound. Among
                  reproducible taxonomies the more informative one is preferred,
                  because a 2-way split is perfectly stable and useless.
      local maxima — every k that beats both neighbours, so the curve's shape is
                  visible instead of being collapsed to one number.
    """
    km = km.sort_values("k").reset_index(drop=True)
    lower = km["stability"] - km["stability_sd"]

    peak = km.loc[km["stability"].idxmax()]
    robust = km[lower >= STABLE_THRESHOLD]

    # Within a single sweep, the largest k clearing the threshold. This is a per-run
    # reading and NOT the reported answer: see consensus() for why.
    selected = km.loc[robust["k"].idxmax()] if len(robust) else peak

    s = km["stability"].to_numpy()
    local_maxima = [int(km["k"].iloc[i]) for i in range(len(s))
                    if (i == 0 or s[i] > s[i - 1])
                    and (i == len(s) - 1 or s[i] > s[i + 1])]

    monotone = bool(km["stability"].is_monotonic_decreasing)
    exhausted = bool(len(robust) and robust["k"].max() == km["k"].max())
    return {
        "rule": ("PER-RUN reading only: largest k whose stability minus one standard "
                 f"deviation stays at or above {STABLE_THRESHOLD}. The reported answer "
                 "is the cross-run consensus — see consensus() and --consensus."),
        "peak_k": int(peak["k"]),
        "peak_stability": round(float(peak["stability"]), 4),
        "selected_k": int(selected["k"]),
        "selected_stability": round(float(selected["stability"]), 4),
        "selected_sd": round(float(selected["stability_sd"]), 4),
        "selected_lower_bound": round(float(selected["stability"]
                                            - selected["stability_sd"]), 4),
        "threshold": STABLE_THRESHOLD,
        "robust_k_values": [int(v) for v in robust["k"]],
        "local_maxima": local_maxima,
        "curve_is_monotonic_decreasing": monotone,
        "threshold_never_crossed": exhausted,
        "caveat": (
            "stability declines monotonically, so the peak is uninformative and the "
            "threshold rule carries the decision" if monotone else None
        ),
    }


def report(result, stem, meta):
    """Print the curve, choose k, and write the CSV and JSON."""
    km = result[result.algorithm == "kmeans"].dropna(subset=["stability"])
    choice = select_k(km)

    print("\nStability curve (KMeans):")
    print(km[["param", "stability", "stability_sd", "null_stability", "margin"]]
          .to_string(index=False))
    print(f"\n  peak:      k={choice['peak_k']} "
          f"(stability {choice['peak_stability']:.4f})")
    print(f"  selected:  k={choice['selected_k']} "
          f"(stability {choice['selected_stability']:.4f} "
          f"+/- {choice['selected_sd']:.4f}, "
          f"lower bound {choice['selected_lower_bound']:.4f})")
    print(f"  local maxima at k = {choice['local_maxima']}")
    print(f"  robust (lower bound >= {STABLE_THRESHOLD}): "
          f"k = {choice['robust_k_values']}")
    if choice["caveat"]:
        print(f"  caveat: {choice['caveat']}")
    if choice["threshold_never_crossed"]:
        print(f"  warning: the sweep may need to extend past "
              f"k={K_RANGE.stop - 1}")

    summary = {**meta, "selection": choice,
               "curve": [
                   {"k": int(r.k), "stability": round(r.stability, 4),
                    "sd": round(r.stability_sd, 4),
                    "null": round(r.null_stability, 4),
                    "margin": round(r.margin, 4)}
                   for r in km.itertuples()]}
    (REPORTS / f"{stem}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {stem}.json")
    return choice


def consensus(stems, threshold=STABLE_THRESHOLD):
    """Decide k by what replicates across independent sweeps.

    A single stability curve is noisier than it looks. Three sweeps on this corpus —
    10,000, 20,000 and 50,000 abstracts — each produce a clean-looking curve, and they
    disagree about which k are stable:

        pilot  n=10,000   clears at k = 3, 4, 5, 8          (dips at 6 and 7)
        main   n=20,000   clears at k = 3, 4, 5, 6, 7, 8    (collapses from 9)
        main   n=50,000   clears at k = 3, 5, 6, 7, 8, 9, 15, 16, 17, 18

    No shape-based rule survives all three. "Largest k clearing the threshold" returns
    18 on the third run, walking straight past the collapse into a revival that does
    not replicate. "Largest k before the first sustained collapse" returns 5 on the
    pilot, because the pilot's real dip at k=6..7 looks like a cliff. Both rules were
    tried; both were discarded, and discarding them is recorded in the report.

    What survives is replication. A number of categories is only meaningful if an
    independent sample of the same corpus finds it again, which is the premise the
    whole method rests on — applied to the sweeps themselves rather than only to the
    subsamples inside one sweep. So: intersect the stable sets, take the largest.

    This is strictly more conservative than any single-run rule. It cannot select a k
    that any run found unstable, and it is insensitive to where a given curve happens
    to be ragged.
    """
    per_run, stable_sets = {}, []
    for stem in stems:
        csv = REPORTS / f"{stem}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} missing — run the sweep first")
        d = pd.read_csv(csv)
        d = d[d["algorithm"] == "kmeans"].sort_values("k")
        lower = d["stability"] - d["stability_sd"]
        stable = sorted(int(v) for v in d["k"][lower >= threshold])
        n = int(d["n_papers"].iloc[0]) if "n_papers" in d else None
        per_run[stem] = {"n_papers": n, "stable_k": stable}
        stable_sets.append(set(stable))

    shared = sorted(set.intersection(*stable_sets)) if stable_sets else []
    return {
        "rule": (f"largest k whose one-standard-deviation lower bound clears "
                 f"{threshold} in EVERY independent sweep"),
        "runs": per_run,
        "stable_in_every_run": shared,
        "selected_k": max(shared) if shared else None,
        "n_runs": len(stems),
    }


def rescore(stem):
    """Re-apply the selection rule to a sweep that has already been run."""
    csv = REPORTS / f"{stem}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} missing — run the sweep first")
    old = REPORTS / f"{stem}.json"
    meta = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    meta.pop("selection", None)
    meta.pop("curve", None)
    report(pd.read_csv(csv), stem, meta)


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
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the selection from a saved sweep, no clustering")
    ap.add_argument("--rescore-stem", metavar="STEM",
                    help="which saved sweep --rescore should re-read")
    ap.add_argument("--consensus", nargs="*", metavar="STEM",
                    help="decide k from what replicates across saved sweeps; with no "
                         "arguments, uses every stability_*_embedding*.csv in reports/")
    args = ap.parse_args()

    if args.consensus is not None:
        stems = args.consensus or sorted(
            p.stem for p in REPORTS.glob("stability_*_embedding*.csv"))
        result = consensus(stems)
        for stem, info in result["runs"].items():
            print(f"  {stem}\n      n={info['n_papers']}  "
                  f"stable at k = {info['stable_k']}")
        print(f"\n  stable in all {result['n_runs']} runs: "
              f"{result['stable_in_every_run']}")
        print(f"  selected k = {result['selected_k']}")
        out = REPORTS / "stability_consensus.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {out.name}")
        return

    if args.rescore:
        if not args.rescore_stem:
            raise SystemExit(
                "--rescore needs --rescore-stem, e.g. "
                "--rescore-stem stability_main_minilm_embedding_n50000")
        rescore(args.rescore_stem)
        return

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

    # The sample size is part of the filename. Two sweeps differing only in --max-n are
    # different experiments, and an earlier build let the second silently overwrite the
    # first, leaving one run's curve stamped with the other's metadata.
    stem = f"stability_{args.phase}_{args.model}_{args.space}_n{n_used}"

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
    def tabulate(rows):
        result = pd.DataFrame(rows)
        for i, (col, val) in enumerate([("phase", args.phase), ("model", args.model),
                                        ("space", args.space), ("n_papers", n_used),
                                        ("reps", args.reps)]):
            result.insert(i, col, val)
        return result

    # Persist after every configuration: the sweep takes hours and an interrupted
    # run should keep what it finished. --rescore then works on the partial sweep.
    csv_path = REPORTS / f"{stem}.csv"
    rows = sweep(X, X_null, configs, args.reps,
                 on_row=lambda rs: tabulate(rs).to_csv(csv_path, index=False))

    result = tabulate(rows)
    result.to_csv(csv_path, index=False)

    report(result, stem, {
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
        "null_model": ("uniform on unit sphere" if on_sphere
                       else "uniform in per-dimension bounding box"),
    })


if __name__ == "__main__":
    main()
