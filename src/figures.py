"""Generate every figure the report uses, from artifacts already on disk.

Each figure is skipped with a message if its inputs are missing, so this can be run
at any point in the pipeline and re-run as results land.

Design rules applied throughout
-------------------------------
* Categorical hues are assigned in a fixed order and never cycled. Three of the hues
  sit below 3:1 contrast on the chart surface, which obliges visible direct labels —
  so every series and every cluster is labelled in place rather than identified by
  colour alone.
* The UMAP scatter is drawn as small multiples, one panel per category, because a
  single scatter in eight colours cannot be told apart under colour-vision deficiency
  (adjacent-pair separation holds for lines and bars, but a scatter puts all pairs on
  screen at once). The combined overview is kept only as a navigational view, with
  every cluster labelled at its centroid.
* Sequential encodings (heatmaps) use one hue, light to dark. No rainbow maps.
* One y-axis per panel, never two scales in one frame.

Usage:   python figures.py [--phase pilot|main] [--model minilm|mpnet]
                           [--labels hdbscan_mcs100]
Outputs: reports/figures/*.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from cluster import UMAP_DIMS
from config import FIGURES, REPORTS, SEED, work_path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e4e3de"

# Fixed categorical order. Never cycled: a ninth series folds into "other".
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
PRIMARY = CATEGORICAL[0]
# Single-hue sequential ramp, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#f2f7fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"])

DPI = 160


def style(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    """Recessive grid and axes; the data carries the ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12,
                     fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    return ax


def figure(w, h, n=1, **kw):
    fig, axes = plt.subplots(1, n, figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, axes


def save(fig, name, note=None):
    FIGURES.mkdir(parents=True, exist_ok=True)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=INK_MUTED, va="bottom")
    fig.savefig(FIGURES / name, dpi=DPI, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def load_json(name):
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def spread_labels(points, extent, min_gap=0.11, iterations=80):
    """Nudge overlapping label anchors apart so no two collide.

    Labels sit at cluster centroids, and two adjacent clusters can have centroids
    close enough that their text boxes overlap. A few rounds of pairwise repulsion
    separate them while keeping each label near the cluster it names.
    """
    pts = np.asarray(points, dtype=float).copy()
    if len(pts) < 2:
        return pts
    gap = min_gap * extent
    for _ in range(iterations):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                delta = pts[i] - pts[j]
                dist = float(np.hypot(*delta))
                if dist >= gap:
                    continue
                if dist < 1e-9:
                    delta, dist = np.array([gap, 0.0]), gap
                push = (gap - dist) / 2 * delta / dist
                pts[i] += push
                pts[j] -= push
                moved = True
        if not moved:
            break
    return pts


# --- EDA -----------------------------------------------------------------------

def fig_papers_per_year(eda):
    per_year = {int(y): n for y, n in eda["per_year"].items()}
    years = sorted(y for y in per_year if 1991 <= y <= 2026)
    counts = [per_year[y] for y in years]

    fig, ax = figure(9, 4)
    kept = [y >= 2016 for y in years]
    ax.bar([y for y, k in zip(years, kept) if not k],
           [c for c, k in zip(counts, kept) if not k],
           color="#c9c8c2", width=0.75, zorder=3, label="excluded (pre-2016)")
    ax.bar([y for y, k in zip(years, kept) if k],
           [c for c, k in zip(counts, kept) if k],
           color=PRIMARY, width=0.75, zorder=3, label="sampled from (2016+)")

    peak = years[int(np.argmax(counts))]
    ax.annotate(f"{max(counts):,} in {peak}",
                xy=(peak, max(counts)), xytext=(0, 8),
                textcoords="offset points", ha="center",
                fontsize=9, color=INK_2)
    ax.text(2016, max(counts) * 0.52, " 2016 cutoff", color=INK_2, fontsize=9,
            va="center")
    ax.axvline(2015.5, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=4)

    style(ax, "arXiv submissions per year",
          ylabel="papers")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    save(fig, "eda_papers_per_year.png",
         f"{eda['total_records']:,} records in the v309 snapshot.")


def fig_archive_distribution(mix):
    agg = {}
    for year, counts in mix["by_year_archive"].items():
        if int(year) >= 2016:
            for a, n in counts.items():
                agg[a] = agg.get(a, 0) + n
    s = pd.Series(agg).sort_values()
    total = s.sum()

    fig, ax = figure(8, 7)
    ax.barh(s.index, s.values, color=PRIMARY, height=0.72, zorder=3)
    for name, v in s.items():
        ax.text(v + total * 0.004, name, f"{v:,}  ({100 * v / total:.1f}%)",
                va="center", fontsize=8.5, color=INK_2)

    style(ax, "Primary archive distribution, 2016 onward", grid_axis="x")
    ax.set_xlim(0, s.max() * 1.22)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    save(fig, "eda_archive_distribution.png",
         f"All {len(s)} archives are sampled, with no minimum-share cutoff: a 1% "
         f"cutoff would delete econ and q-fin entirely.")


def fig_abstract_length(eda):
    words = {int(w): n for w, n in eda["abstract_words"].items()}
    xs = np.arange(0, 401)
    ys = np.array([words.get(x, 0) for x in xs])

    fig, ax = figure(9, 4)
    ax.fill_between(xs, ys, color=PRIMARY, alpha=0.18, zorder=2)
    ax.plot(xs, ys, color=PRIMARY, linewidth=2, zorder=3)

    # 256 wordpiece tokens is roughly 190 words for this corpus.
    for x, label, colour in ((20, "20-word minimum", INK_2),
                             (190, "~256 tokens", INK_2)):
        ax.axvline(x, color=colour, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
        ax.text(x + 4, ys.max() * 0.9, label, fontsize=9, color=colour)

    style(ax, "Abstract length", xlabel="words", ylabel="papers")
    ax.set_xlim(0, 400)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    save(fig, "eda_abstract_length.png",
         "Truncating at 256 tokens keeps 67.2% of abstracts whole; 128 would keep "
         "only 20.5%.")


# --- Clustering ----------------------------------------------------------------

def fig_k_sweep(phase, model):
    path = REPORTS / f"clustering_{phase}_{model}.csv"
    if not path.exists():
        print(f"  skip k-sweep: {path.name} not found")
        return
    d = pd.read_csv(path)
    km = d[d.algorithm == "kmeans"].sort_values("n_clusters")

    fig, (ax1, ax2) = figure(11, 4.4, n=2)

    series = [("silhouette", "silhouette", CATEGORICAL[0]),
              ("ari_archive", "ARI vs 20 archives", CATEGORICAL[1]),
              ("ami_group", "AMI vs 8 groups", CATEGORICAL[2])]
    for col, label, colour in series:
        ax1.plot(km.n_clusters, km[col], color=colour, linewidth=2,
                 marker="o", markersize=4.5, zorder=3)
        # Peak marked with a ring; the k value alone, since the series is named
        # by its end-of-line label rather than repeated here.
        peak = km.loc[km[col].idxmax()]
        ax1.scatter([peak.n_clusters], [peak[col]], s=110, facecolors="none",
                    edgecolors=colour, linewidths=1.6, zorder=4)
        ax1.annotate(f"peak k={int(peak.n_clusters)}",
                     xy=(peak.n_clusters, peak[col]), xytext=(0, 11),
                     textcoords="offset points", ha="center",
                     fontsize=8.5, color=INK_2)

    # Direct labels at the right end: a coloured dot carries identity, the text
    # stays in ink, so no legend box is needed.
    k_max = int(km.n_clusters.max())
    for col, label, colour in series:
        y = float(km[col].iloc[-1])
        ax1.scatter([k_max + 0.6], [y], s=26, color=colour, zorder=5,
                    clip_on=False)
        ax1.text(k_max + 1.1, y, label, fontsize=9, color=INK_2,
                 va="center", clip_on=False)

    style(ax1, "KMeans: internal and external scores disagree",
          xlabel="number of clusters (k)", ylabel="score")
    ax1.set_xlim(km.n_clusters.min() - 0.6, k_max + 5.4)
    ax1.set_xticks(range(int(km.n_clusters.min()), k_max + 1, 2))
    lo = min(km[c].min() for c, _, _ in series)
    hi = max(km[c].max() for c, _, _ in series)
    ax1.set_ylim(lo - 0.03 * (hi - lo), hi + 0.14 * (hi - lo))  # room for peak labels

    ax2.plot(km.n_clusters, km.largest_cluster_pct, color=CATEGORICAL[7],
             linewidth=2, marker="o", markersize=4.5, zorder=3)
    ax2.axhline(25, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax2.text(k_max, 25.9, "a quarter of the corpus", fontsize=8.5,
             color=INK_2, ha="right")
    worst = km.loc[km.largest_cluster_pct.idxmax()]
    ax2.annotate(f"k={int(worst.n_clusters)}: one cluster holds\n"
                 f"{worst.largest_cluster_pct:.0f}% of the corpus",
                 xy=(worst.n_clusters, worst.largest_cluster_pct),
                 xytext=(20, 4), textcoords="offset points",
                 fontsize=8.5, color=INK_2)
    style(ax2, "The silhouette optimum is degenerate",
          xlabel="number of clusters (k)", ylabel="largest cluster (% of corpus)")
    ax2.set_xticks(range(int(km.n_clusters.min()), k_max + 1, 2))
    ax2.set_ylim(0, max(km.largest_cluster_pct) * 1.12)

    save(fig, f"clustering_k_sweep_{phase}.png",
         "Silhouette prefers k=3-4, where a single cluster holds 44% of all papers. "
         "This is why the number of categories is settled by stability, not silhouette.")


def fig_hdbscan(phase, model):
    path = REPORTS / f"clustering_{phase}_{model}.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    hdb = d[d.algorithm == "hdbscan"].copy()
    if hdb.empty:
        return
    hdb["mcs"] = hdb["param"].str.split("=").str[1].astype(int)
    hdb = hdb.sort_values("mcs")

    fig, (ax1, ax2) = figure(11, 4.4, n=2)
    x = np.arange(len(hdb))
    labels = [f"{m:,}" for m in hdb.mcs]

    ax1.bar(x - 0.2, hdb.dbcv, width=0.38, color=CATEGORICAL[0], zorder=3)
    ax1.bar(x + 0.2, hdb.ari_archive, width=0.38, color=CATEGORICAL[1], zorder=3)
    for xi, (dbcv, ari) in enumerate(zip(hdb.dbcv, hdb.ari_archive)):
        ax1.text(xi - 0.2, dbcv + 0.012, f"{dbcv:.3f}", ha="center",
                 fontsize=8, color=INK_2)
        ax1.text(xi + 0.2, ari + 0.012, f"{ari:.3f}", ha="center",
                 fontsize=8, color=INK_2)
    ax1.set_xticks(x, labels)
    style(ax1, "HDBSCAN after UMAP reduction",
          xlabel="min_cluster_size", ylabel="score")
    ax1.legend(["DBCV (internal)", "ARI vs archives (external)"], frameon=False,
               fontsize=8.5, labelcolor=INK_2, loc="upper right")

    ax2.plot(x, hdb.noise_pct, color=CATEGORICAL[7], linewidth=2, marker="o",
             markersize=5, zorder=3)
    for xi, (n, c) in enumerate(zip(hdb.noise_pct, hdb.n_clusters)):
        ax2.annotate(f"{int(c)} clusters\n{n:.1f}% noise", xy=(xi, n),
                     xytext=(0, 9), textcoords="offset points", ha="center",
                     fontsize=8, color=INK_2)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, max(hdb.noise_pct) * 1.45)
    style(ax2, "Noise stays low where Arcan reported up to 74%",
          xlabel="min_cluster_size", ylabel="unassigned (%)")

    save(fig, f"clustering_hdbscan_{phase}.png",
         "Reducing with UMAP before density clustering is what prevents the collapse "
         "Arcan (2026) reported for HDBSCAN.")


def fig_stability(phase, model, space="embedding"):
    path = REPORTS / f"stability_{phase}_{model}_{space}.json"
    if not path.exists():
        print(f"  skip stability: {path.name} not found")
        return
    s = json.loads(path.read_text(encoding="utf-8"))
    curve = pd.DataFrame(s["curve"])

    fig, ax = figure(9, 4.6)
    ax.fill_between(curve.k, curve.stability - curve.sd,
                    curve.stability + curve.sd,
                    color=CATEGORICAL[0], alpha=0.16, zorder=2, linewidth=0)
    ax.plot(curve.k, curve.stability, color=CATEGORICAL[0], linewidth=2.2,
            marker="o", markersize=5, zorder=4)
    ax.plot(curve.k, curve["null"], color=INK_MUTED, linewidth=1.8,
            linestyle=(0, (5, 3)), zorder=3)

    ax.text(curve.k.iloc[-1], curve.stability.iloc[-1], "  observed",
            color=CATEGORICAL[0], fontsize=9.5, va="center", fontweight="bold")
    ax.text(curve.k.iloc[-1], curve["null"].iloc[-1], "  null",
            color=INK_MUTED, fontsize=9.5, va="center")

    sel = s["selection"]
    ax.axhline(sel["threshold"], color=INK_2, linewidth=1,
               linestyle=(0, (2, 3)), zorder=2)
    ax.text(curve.k.min(), sel["threshold"] + 0.012,
            f"stable threshold {sel['threshold']}", fontsize=8.5, color=INK_2)

    chosen = sel["largest_stable_k"]
    row = curve[curve.k == chosen]
    if len(row):
        ax.scatter(row.k, row.stability, s=150, facecolors="none",
                   edgecolors=INK, linewidths=1.6, zorder=5)
        ax.annotate(f"k={chosen}\nmost granular stable solution",
                    xy=(chosen, float(row.stability.iloc[0])),
                    xytext=(10, -26), textcoords="offset points",
                    fontsize=9, color=INK)

    style(ax, f"Cluster stability in {space} space",
          xlabel="number of clusters (k)",
          ylabel="ARI between independent subsamples")
    ax.set_xticks(list(curve.k))
    save(fig, f"stability_{phase}_{space}.png",
         f"{s['reps']} subsample pairs per k on {s['n_papers']:,} papers. "
         f"Null: {s['null_model']}. No arXiv label is consulted.")


# --- The discovered categories --------------------------------------------------

def umap_2d(phase, model, cap, force=False):
    """A 2D projection for plotting only. Clustering uses the 50d reduction."""
    out = work_path(phase, f"umap_{model}_2d_{cap}.npy")
    idx_file = work_path(phase, f"umap_{model}_2d_{cap}_idx.npy")
    if out.exists() and idx_file.exists() and not force:
        return np.load(out), np.load(idx_file)

    source = work_path(phase, f"embeddings_{model}.npy")
    if not source.exists():
        return None, None
    X = np.load(source)
    rng = np.random.default_rng(SEED)
    idx = (np.sort(rng.choice(len(X), cap, replace=False))
           if len(X) > cap else np.arange(len(X)))

    import umap
    print(f"  computing 2D UMAP on {len(idx):,} papers (plot only)...", flush=True)
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=SEED
                   ).fit_transform(X[idx]).astype(np.float32)
    np.save(out, xy)
    np.save(idx_file, idx)
    return xy, idx


def fig_umap(phase, model, labels_config, cap=25_000):
    label_file = work_path(phase, f"labels_{model}_{labels_config}.npy")
    if not label_file.exists():
        print(f"  skip UMAP scatter: {label_file.name} not found")
        return
    xy, idx = umap_2d(phase, model, cap)
    if xy is None:
        return

    labels = np.load(label_file)[idx]
    df = pd.read_parquet(work_path(phase, "sample.parquet"),
                         columns=["primary_group"]).iloc[idx]
    groups = df["primary_group"].to_numpy()

    clusters = sorted(c for c in set(labels) if c >= 0)
    group_names = sorted(set(groups))

    # --- overview: two panels, every category labelled at its centroid ----------
    fig, (ax1, ax2) = figure(13, 6.2, n=2)
    for ax, lab, keys, title in (
            (ax1, labels, clusters, "Discovered categories"),
            (ax2, groups, group_names, "arXiv's 8 top-level groups")):
        noise = np.array([l == -1 for l in lab])
        if noise.any():
            ax.scatter(xy[noise, 0], xy[noise, 1], s=1.5, c="#dddcd6",
                       linewidths=0, zorder=1)
        anchors = []
        for i, key in enumerate(keys):
            m = lab == key
            ax.scatter(xy[m, 0], xy[m, 1], s=1.6,
                       c=CATEGORICAL[i % len(CATEGORICAL)],
                       linewidths=0, alpha=0.55, zorder=2)
            anchors.append((np.median(xy[m, 0]), np.median(xy[m, 1])))

        extent = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
        for (cx, cy), key in zip(spread_labels(anchors, extent), keys):
            name = f"cluster {key}" if isinstance(key, (int, np.integer)) else key
            ax.text(cx, cy, name, fontsize=9.5, fontweight="bold",
                    color=INK, ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                              edgecolor=GRID, linewidth=0.8, alpha=0.88))
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10,
                     fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)

    save(fig, f"umap_overview_{phase}.png",
         "Every category is labelled in place, so identity never depends on colour "
         "alone. Grey points are abstracts HDBSCAN left unassigned.")

    # --- small multiples: one panel per discovered category --------------------
    n = len(clusters)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, clusters):
        m = labels == c
        ax.scatter(xy[~m, 0], xy[~m, 1], s=1.0, c="#e2e1db", linewidths=0,
                   zorder=1)
        ax.scatter(xy[m, 0], xy[m, 1], s=1.6, c=PRIMARY, linewidths=0, zorder=2)
        ax.set_title(f"cluster {c}  ·  {m.sum():,} papers  "
                     f"({100 * m.mean():.1f}%)",
                     fontsize=9.5, color=INK, loc="left", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Each discovered category in the shared embedding space",
                 fontsize=12.5, color=INK, x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, f"umap_facets_{phase}.png",
         "One hue per panel: a single eight-colour scatter cannot be separated "
         "reliably under colour-vision deficiency.")


def fig_composition(phase, model):
    path = REPORTS / f"cluster_profiles_{phase}_{model}.md"
    comp = REPORTS / f"cluster_composition_{phase}_{model}.csv"
    if not comp.exists():
        print("  skip composition heatmap: run interpret.py first")
        return

    # Rebuild the contingency table from the markdown report's committed source.
    labels_json = load_json(f"cluster_interpretation_{phase}_{model}.json")
    if not labels_json:
        return
    label_file = work_path(phase, labels_json["label_file"])
    if not label_file.exists():
        return
    labels = np.load(label_file)
    df = pd.read_parquet(work_path(phase, "sample.parquet"),
                         columns=["primary_archive"])
    mask = labels >= 0
    tab = pd.crosstab(labels[mask], df.loc[mask, "primary_archive"])
    # Row-normalise: the question is what each cluster is made of.
    pct = 100 * tab.div(tab.sum(axis=1), axis=0)

    fig, ax = figure(max(9, 0.52 * pct.shape[1] + 3), 0.52 * pct.shape[0] + 2.6)
    im = ax.imshow(pct.values, cmap=SEQUENTIAL, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(pct.shape[1]), pct.columns, rotation=45, ha="right",
                  fontsize=9)
    ax.set_yticks(range(pct.shape[0]), [f"cluster {c}" for c in pct.index],
                  fontsize=9)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.values[i, j]
            if v >= 4:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 55 else INK)
    ax.tick_params(colors=INK_2, length=0)
    ax.set_title("What each discovered category is made of",
                 color=INK, fontsize=12, loc="left", pad=12,
                 fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("% of the cluster", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, length=0, labelsize=8)
    cb.outline.set_visible(False)

    save(fig, f"cluster_archive_heatmap_{phase}.png",
         "Rows sum to 100%. The discovered categories merge all of computer science, "
         "statistics and economics into one, while splitting physics into several — "
         "the opposite of how arXiv organises its groups.")


def fig_confusion(phase, labels_config):
    """Only produced once classifier results exist (local run or Colab)."""
    candidates = sorted(REPORTS.glob(f"classifier_results_{phase}_*.json")) + \
                 sorted(REPORTS.glob("classifier_*.json"))
    best = None
    for path in candidates:
        d = json.loads(path.read_text(encoding="utf-8"))
        block = d.get("best") or d.get("results", {}).get("test")
        cm = (block or {}).get("confusion_matrix")
        if cm:
            best = (d, np.array(cm))
            break
    if best is None:
        print("  skip confusion matrix: no classifier results yet")
        return
    d, cm = best

    pct = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = cm.shape[0]
    fig, ax = figure(0.68 * n + 4.2, 0.62 * n + 3.4)
    im = ax.imshow(pct, cmap=SEQUENTIAL, vmin=0, vmax=100)
    names = [f"c{i}" for i in range(n)]
    ax.set_xticks(range(n), names, fontsize=9)
    ax.set_yticks(range(n), names, fontsize=9)
    for i in range(n):
        for j in range(n):
            if pct[i, j] >= 1:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if pct[i, j] > 55 else INK)
    ax.tick_params(colors=INK_2, length=0)
    ax.set_xlabel("predicted category", color=INK_2, fontsize=10)
    ax.set_ylabel("true category", color=INK_2, fontsize=10)
    ax.set_title("Classifier confusion on the test split", color=INK,
                 fontsize=12, loc="left", pad=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("% of true category", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, length=0, labelsize=8)
    cb.outline.set_visible(False)
    save(fig, f"confusion_matrix_{phase}.png", "Rows sum to 100%.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "main"], default="pilot")
    ap.add_argument("--model", choices=["minilm", "mpnet"], default="minilm")
    ap.add_argument("--labels", default="hdbscan_mcs100")
    ap.add_argument("--scatter-cap", type=int, default=25_000)
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"\nFigures for {args.phase}/{args.model} -> {FIGURES}")

    eda, mix = load_json("eda_stats.json"), load_json("year_mix.json")
    if eda:
        fig_papers_per_year(eda)
        fig_abstract_length(eda)
    if mix:
        fig_archive_distribution(mix)

    fig_k_sweep(args.phase, args.model)
    fig_hdbscan(args.phase, args.model)
    fig_stability(args.phase, args.model, "embedding")
    fig_stability(args.phase, args.model, "umap")
    fig_composition(args.phase, args.model)
    fig_umap(args.phase, args.model, args.labels, args.scatter_cap)
    fig_confusion(args.phase, args.labels)

    print("\nDone.")


if __name__ == "__main__":
    main()
