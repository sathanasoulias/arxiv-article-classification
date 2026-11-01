"""Describe what each discovered cluster is actually about.

A number of clusters is not an answer to "report the best possible number of
categories" — a category has to mean something. This step turns each cluster into a
readable profile built from three independent sources of evidence:

  1. c-TF-IDF terms   — the words that distinguish this cluster from the others
                         (Grootendorst, "BERTopic", arXiv 2022, eq. 1). Plain TF-IDF
                         per document would surface words common to every abstract;
                         treating each cluster as one aggregated document surfaces
                         what is specific to it.
  2. Representative abstracts — the papers nearest the cluster centroid, so the
                         profile can be sanity-checked by reading.
  3. arXiv composition — which archives and groups land in this cluster. This is
                         where the reference taxonomy belongs: as a device for
                         *interpreting* a cluster, not for choosing or scoring one.

Homogeneity and completeness are reported separately rather than only as V-measure,
because they fail in opposite directions and the distinction is the finding: a cluster
set can be pure but fragmented (high homogeneity, low completeness) or coherent but
conflated (the reverse).

Input:   WORK_DIR/<phase>/umap_<model>_50d.npy, sample.parquet
Outputs: WORK_DIR/<phase>/labels_<model>_<config>.npy   (reused by the classifier)
         reports/cluster_profiles_<phase>_<model>_<config>.md
         reports/cluster_composition_<phase>_<model>_<config>.csv
Usage:   python interpret.py --phase pilot --model minilm --algorithm kmeans --k 7
         python interpret.py --phase main --model minilm --algorithm hdbscan --mcs 1500
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    completeness_score, homogeneity_score, v_measure_score,
)

from cluster import UMAP_DIMS
from config import REPORTS, SEED, work_path

TOP_TERMS = 12
N_EXAMPLES = 3
MIN_DF = 10

# Words that appear across scientific abstracts regardless of field. Left in the
# default English stop list they would dominate every cluster's top terms and make the
# profiles unreadable.
BOILERPLATE = {
    "paper", "papers", "article", "study", "studies", "research", "work", "works",
    "results", "result", "show", "shows", "shown", "propose", "proposed", "proposes",
    "present", "presents", "presented", "approach", "approaches", "method", "methods",
    "methodology", "using", "used", "use", "uses", "based", "new", "novel", "also",
    "two", "three", "one", "obtained", "given", "consider", "considered", "find",
    "found", "respectively", "well", "may", "can", "however", "thus", "therefore",
    "finally", "introduce", "introduced", "provide", "provides", "data", "set", "sets",
    "case", "cases", "number", "numbers", "time", "times", "different", "various",
    "first", "second", "general", "particular", "due", "within", "via", "le", "non",
}
TOKEN = re.compile(r"(?u)\b[a-z][a-z-]{2,}\b")   # drops digits and LaTeX fragments


def load_space(phase, model, space, algorithm):
    """Load the matrix the final partition is fitted in.

    KMeans is fitted in the same space where k was chosen (stability.py works on the
    raw L2-normalised embeddings), because selecting a number in one space and then
    producing the partition in another is not a claim the evidence supports.

    HDBSCAN always uses the UMAP reduction: density estimates degrade in high
    dimensions, and without the reduction it collapses — which is exactly the failure
    Arcan (2026) reported.
    """
    if algorithm == "hdbscan" and space != "umap":
        print("  note: HDBSCAN requires the UMAP reduction; using umap space")
        space = "umap"
    path = (work_path(phase, f"umap_{model}_{UMAP_DIMS}d.npy") if space == "umap"
            else work_path(phase, f"embeddings_{model}.npy"))
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    return np.load(path), space


def fit_clustering(X, algorithm, k, mcs):
    """Re-fit the selected configuration. Deterministic, so labels match cluster.py."""
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        return model.fit_predict(X)
    return HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(X)


def ctfidf(texts, labels, top_n=TOP_TERMS):
    """c-TF-IDF: terms that distinguish each cluster from the rest of the corpus.

    W(t,c) = tf(t,c) * log(1 + A / f(t)), with A the mean number of words per
    cluster-document and f(t) the term's total frequency across all clusters.
    """
    stop = list(CountVectorizer(stop_words="english").get_stop_words() | BOILERPLATE)
    vec = CountVectorizer(
        stop_words=stop, token_pattern=TOKEN.pattern, min_df=MIN_DF,
        max_df=0.5,          # a term in over half the abstracts cannot distinguish
    )
    counts = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    clusters = sorted(c for c in set(labels) if c >= 0)
    # Aggregate each cluster into a single document.
    per_cluster = np.vstack([
        np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in clusters
    ])

    tf = per_cluster / np.maximum(per_cluster.sum(axis=1, keepdims=True), 1)
    total = np.maximum(per_cluster.sum(axis=0), 1)
    idf = np.log(1 + per_cluster.sum(axis=1).mean() / total)
    weights = tf * idf

    return {
        c: vocab[np.argsort(-weights[i])[:top_n]].tolist()
        for i, c in enumerate(clusters)
    }


def representatives(X, labels, df, n=N_EXAMPLES):
    """Papers nearest each cluster centroid — the readable sanity check."""
    out = {}
    for c in sorted(s for s in set(labels) if s >= 0):
        member = np.flatnonzero(labels == c)
        centroid = X[member].mean(axis=0)
        nearest = member[np.argsort(
            np.linalg.norm(X[member] - centroid, axis=1))[:n]]
        out[c] = [
            {"id": df.at[i, "id"], "title": df.at[i, "title"],
             "primary": df.at[i, "primary_archive"]}
            for i in nearest
        ]
    return out


def composition(labels, df):
    """Cluster x archive and cluster x group cross-tabs, plus per-cluster purity."""
    mask = labels >= 0
    tab_archive = pd.crosstab(labels[mask], df.loc[mask, "primary_archive"])
    tab_group = pd.crosstab(labels[mask], df.loc[mask, "primary_group"])

    rows = []
    for c in tab_archive.index:
        a, g = tab_archive.loc[c], tab_group.loc[c]
        rows.append({
            "cluster": int(c),
            "n_papers": int(a.sum()),
            "pct_of_corpus": round(100 * a.sum() / len(df), 1),
            "top_archive": a.idxmax(),
            "top_archive_pct": round(100 * a.max() / a.sum(), 1),
            "top_group": g.idxmax(),
            "top_group_pct": round(100 * g.max() / g.sum(), 1),
            "archives_over_10pct": ", ".join(
                a[a / a.sum() > 0.10].sort_values(ascending=False).index),
        })
    return pd.DataFrame(rows), tab_archive, tab_group


def agreement(labels, df):
    """Homogeneity and completeness against both reference levels.

    Reported for interpretation only. These never influenced the choice of k — that
    was settled in stability.py without reference to any arXiv label.
    """
    mask = labels >= 0
    out = {}
    for level in ("primary_archive", "primary_group"):
        truth = df.loc[mask, level]
        out[level] = {
            "homogeneity": round(homogeneity_score(truth, labels[mask]), 4),
            "completeness": round(completeness_score(truth, labels[mask]), 4),
            "v_measure": round(v_measure_score(truth, labels[mask]), 4),
        }
    return out


def write_report(path, meta, terms, reps, comp, scores, tab_archive):
    lines = [
        f"# Discovered categories — {meta['phase']}/{meta['model']}, "
        f"{meta['config']} ({meta['space']} space)",
        "",
        f"{meta['n_clusters']} clusters over {meta['n_papers']:,} abstracts"
        + (f", {meta['noise_pct']}% left unassigned as noise."
           if meta["noise_pct"] else "."),
        "",
        "Cluster numbers below are arbitrary identifiers. Names are the author's "
        "reading of the c-TF-IDF terms and representative titles, not model output.",
        "",
        "## Agreement with the arXiv taxonomy",
        "",
        "Interpretation only — no arXiv label was used to select this configuration.",
        "",
        "| reference level | homogeneity | completeness | V-measure |",
        "|---|---|---|---|",
    ]
    for level, s in scores.items():
        label = "20 archives" if "archive" in level else "8 groups"
        lines.append(f"| {label} | {s['homogeneity']:.4f} | "
                     f"{s['completeness']:.4f} | {s['v_measure']:.4f} |")

    lines += ["", "## Cluster profiles", ""]
    for row in comp.itertuples():
        c = row.cluster
        lines += [
            f"### Cluster {c} — {row.n_papers:,} papers "
            f"({row.pct_of_corpus}% of corpus)",
            "",
            f"**Distinguishing terms:** {', '.join(terms.get(c, []))}",
            "",
            f"**Dominant arXiv archive:** `{row.top_archive}` "
            f"({row.top_archive_pct}% of the cluster) · "
            f"**group:** {row.top_group} ({row.top_group_pct}%)",
            "",
            f"**Archives above 10%:** {row.archives_over_10pct}",
            "",
            "**Papers nearest the centroid:**",
            "",
        ]
        for ex in reps.get(c, []):
            lines.append(f"- [{ex['id']}](https://arxiv.org/abs/{ex['id']}) "
                         f"(`{ex['primary']}`) — {ex['title']}")
        lines.append("")

    lines += [
        "## Full cluster x archive contingency table", "",
        tab_archive.to_markdown(), "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--model", choices=["minilm", "mpnet"], default="minilm")
    ap.add_argument("--algorithm", choices=["kmeans", "hdbscan"], default="kmeans")
    ap.add_argument("--k", type=int, default=7, help="kmeans only")
    ap.add_argument("--mcs", type=int, default=100, help="hdbscan min_cluster_size")
    ap.add_argument("--space", choices=["embedding", "umap"], default="embedding",
                    help="where the partition is fitted; hdbscan always uses umap")
    args = ap.parse_args()

    X, space = load_space(args.phase, args.model, args.space, args.algorithm)
    config = (f"kmeans_k{args.k}" if args.algorithm == "kmeans"
              else f"hdbscan_mcs{args.mcs}")
    if space == "embedding":
        config += "_emb"

    df = pd.read_parquet(work_path(args.phase, "sample.parquet"))
    assert len(X) == len(df)

    print(f"\n[{args.phase}/{args.model}] {config} on {len(df):,} abstracts "
          f"in {space} space ({X.shape[1]}d)")
    labels = fit_clustering(X, args.algorithm, args.k, args.mcs)
    n_clusters = len({c for c in labels if c >= 0})
    noise_pct = round(100 * float((labels == -1).mean()), 1)
    print(f"  {n_clusters} clusters, {noise_pct}% noise")

    # Saved so the classifier trains on exactly these targets.
    label_file = work_path(args.phase, f"labels_{args.model}_{config}.npy")
    np.save(label_file, labels)

    print("  c-TF-IDF terms...", flush=True)
    terms = ctfidf(df["abstract"].tolist(), labels)
    reps = representatives(X, labels, df)
    comp, tab_archive, tab_group = composition(labels, df)
    scores = agreement(labels, df)

    meta = {"phase": args.phase, "model": args.model, "config": config,
            "space": space, "algorithm": args.algorithm,
            "n_clusters": n_clusters, "n_papers": len(df), "noise_pct": noise_pct}

    REPORTS.mkdir(exist_ok=True)
    comp.to_csv(REPORTS / f"cluster_composition_{args.phase}_{args.model}_{config}.csv",
                index=False)
    md = REPORTS / f"cluster_profiles_{args.phase}_{args.model}_{config}.md"
    write_report(md, meta, terms, reps, comp, scores, tab_archive)

    (REPORTS / f"cluster_interpretation_{args.phase}_{args.model}_{config}.json").write_text(
        json.dumps({**meta, "seed": SEED, "agreement": scores,
                    "terms": {str(c): t for c, t in terms.items()},
                    "label_file": label_file.name}, indent=2), encoding="utf-8")

    print(f"\n  archives: homogeneity={scores['primary_archive']['homogeneity']:.4f}  "
          f"completeness={scores['primary_archive']['completeness']:.4f}")
    print(f"  groups:   homogeneity={scores['primary_group']['homogeneity']:.4f}  "
          f"completeness={scores['primary_group']['completeness']:.4f}")
    print(f"\nWrote {md.name} and labels_{args.model}_{config}.npy")
    print(comp[["cluster", "n_papers", "top_archive", "top_archive_pct"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
