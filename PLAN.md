# Project plan

Definitive step list for the assignment. Nothing is executed until a step is agreed.

**Confirmed scope (Iris.ai, 2026-10-29):** categories are **discovered from the abstracts**, and
the classifier predicts those discovered categories. arXiv's own subject labels are used only to
*interpret and validate* the clusters — never as the training target. Methodology is ours to choose
and defend.

**Approach:** follow Arcan (2026) — frozen embeddings → clustering → fine-tuned classifier — while
fixing two methodological weaknesses in that paper (Step 5).

---

## 1. Reproducibility rules

Every step obeys these, so any result in the report can be regenerated from a clean checkout.

| Rule | How |
|---|---|
| **One seed** | `SEED = 42` in `config.py`, passed to every sampler, splitter, clusterer and trainer |
| **One source of truth for paths** | All paths come from `config.py`. No hard-coded paths in scripts or notebooks |
| **Cached artifacts** | Each script writes its output to disk and skips the work if the output exists, unless `--force` |
| **Parameters recorded with results** | Every result file embeds the parameters that produced it, so a number is never orphaned from its settings |
| **Scripts, not notebooks, do the work** | `src/*.py` is runnable from the command line. Notebooks only load artifacts and display them |
| **Pinned dependencies** | `requirements.txt` with exact versions, plus the Python version |
| **Documented run order** | README lists the exact commands, in order, to reproduce every figure |
| **Deterministic splits** | Train/val/test computed once, saved, and never recomputed |

Each script gets a module docstring stating its purpose, inputs, outputs and CLI usage.

---

## 2. Repository layout

```
├── src/
│   ├── config.py              # paths, taxonomy mapping, SEED           [done]
│   ├── download_data.py       # fetch the 5.65 GB snapshot from Kaggle  [done]
│   ├── eda_scan.py            # streaming pass → eda_stats.json         [done]
│   ├── eda_plots.py           # eda_stats.json → figures
│   ├── preprocess.py          # clean, stratified sample, fixed splits
│   ├── embed.py               # abstracts → cached embedding matrices
│   ├── cluster.py             # k sweep + HDBSCAN + internal metrics
│   ├── stability.py           # resampling stability across k
│   ├── interpret.py           # c-TF-IDF, examples, arXiv contingency
│   ├── train_classifier.py    # fine-tune a transformer
│   ├── tune.py                # hyperparameter search / model comparison
│   └── evaluate.py            # metrics, confusion matrix, error analysis
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clustering.ipynb
│   └── 03_classification.ipynb
├── data/processed/            # gitignored: sample, splits, embeddings
├── papers/                    # gitignored: reference PDFs
├── reports/
│   ├── eda_stats.json
│   ├── clustering_results.csv
│   ├── stability_results.csv
│   ├── cluster_profiles.md
│   ├── classification_results.csv
│   └── figures/
├── PLAN.md
├── README.md                  # the deliverable report
└── requirements.txt
```

---

## 3. Pipeline at a glance

| Step | Script | Input | Output |
|---|---|---|---|
| 0 | `download_data.py` | Kaggle | 5.65 GB snapshot |
| 1 | — | literature | README §1 |
| 2a | `eda_scan.py` | snapshot | `eda_stats.json` |
| 2b | `eda_plots.py` | `eda_stats.json` | figures |
| 3 | `preprocess.py` | snapshot | `sample.parquet`, `splits.json` |
| 4 | `embed.py` | sample | `embeddings_<model>.npy` |
| 5a | `cluster.py` | embeddings | `clustering_results.csv` |
| 5b | `stability.py` | embeddings | `stability_results.csv` |
| 6 | `interpret.py` | embeddings + labels | `cluster_profiles.md` |
| 7 | `train_classifier.py` | sample + cluster labels | trained model, metrics |
| 8 | `tune.py` | sample + cluster labels | `classification_results.csv` |
| 9 | — | all of the above | README §2–7 |

---

## 4. The steps

### Step 0 — Setup ✅

Dataset downloaded (5.65 GB, cached outside OneDrive), venv created, public repo live.

---

### Step 1 — Literature review ✅

Three core papers: Arcan (2026) for the pipeline, BERTopic (2022) for discovery, Lange et al.
(2004) for choosing the number of clusters. Written into README §1.

**What it established:** Arcan chose k by maximising agreement with arXiv's real labels. We cannot,
because our brief says the number is unknown. That difference is the backbone of our method.

---

### Step 2a — EDA scan ✅

Streaming pass over all 3,224,635 records. Found: zero empty abstracts, median abstract 145 words,
48.3% of papers multi-label, 17.3% spanning disciplines, and severe imbalance — physics 48%, cs 26%,
math 19% (93% between them) against economics at 0.4%.

---

### Step 2b — EDA charts ⬜

**Goal:** turn the counts into the figures the report needs.

**Produces:** abstract-length distribution, class-frequency bar chart (log scale, since the ratio is
118:1), labels-per-paper histogram, papers-per-year growth curve, archive-vs-group comparison.

**Time:** ~10 minutes. No heavy compute — reads the JSON from 2a.

---

### Step 3 — Preprocessing and sampling ⬜

**Goal:** one fixed dataset that every later step reuses.

**What it does:**
1. Stream the snapshot again, keeping `id`, `title`, `abstract`, `categories`, submission year
2. Drop abstracts below a minimum length (the data has a 1-word minimum, so there is a tail to trim)
3. Clean text: strip leading whitespace, collapse newlines, normalise inline LaTeX maths
4. Derive labels at both levels: archive (~20) and group (8)
5. Take a **proportional stratified sample** (~150k) on the primary archive
6. Split 70/15/15 train/val/test, stratified, seeded — **saved once**

**Produces:** `data/processed/sample.parquet`, `data/processed/splits.json`

**Decision it makes:** none left once §5 below is agreed — this step is mechanical.

**Time:** ~8 minutes (one more full pass over 5.65 GB).

---

### Step 4 — Embeddings ⬜

**Goal:** turn each abstract into a vector capturing its meaning.

**What it does:** runs the sampled abstracts through a **frozen** sentence-embedding model and caches
the matrix to disk. Primary model `all-mpnet-base-v2` (best in Arcan's clustering results),
secondary `all-MiniLM-L6-v2` (about 5× faster, used as the comparison that partly satisfies
instruction 6).

**Why frozen:** every paper reviewed uses off-the-shelf embeddings here. Training the encoder is a
separate research project and the evidence suggests no gain.

**Produces:** `data/processed/embeddings_mpnet.npy`, `embeddings_minilm.npy`

**Time on CPU:** MiniLM ~15 min, MPNet ~1 hour. Acceptable; done once.

---

### Step 5a — Clustering and the k sweep ⬜

**The core of the assignment.** Answers *"report the best possible number of categories."*

**What it does:**
1. Reduce embeddings with UMAP (required before density clustering)
2. **KMeans** for every k in 3…20
3. **HDBSCAN**, sweeping `min_cluster_size`
4. Score each configuration:

| Measure | Applies to | Tells us |
|---|---|---|
| Silhouette | KMeans | geometric separation |
| DBCV | HDBSCAN | separation for non-convex clusters (silhouette is invalid here) |
| Noise fraction | HDBSCAN | share of papers left unassigned |
| Cluster-size distribution | both | whether one cluster swallows everything |

**The two fixes to Arcan's method:**

1. **No peeking.** Arcan selected k by maximising `0.5·ARI + 0.5·NMI` against arXiv's true labels.
   We select using internal measures and stability only, then compare against arXiv *afterwards* as
   independent validation. Using the answer to find the answer is circular, and our brief says the
   number is unknown.
2. **UMAP before HDBSCAN.** Arcan ran HDBSCAN on raw high-dimensional vectors and it collapsed
   (3–5 clusters, up to 74% noise). Density clustering degrades in high dimensions; reducing first
   is the standard recipe.

**Produces:** `reports/clustering_results.csv` — one row per configuration with all metrics.

**Time:** ~20 minutes for the full sweep on cached embeddings.

---

### Step 5b — Stability ⬜

**Goal:** the strongest evidence for the chosen number of categories (Lange et al., 2004).

**What it does:** for each k, draw repeated 80% subsamples, cluster each, and measure how much the
groupings agree (ARI between runs). Real structure reproduces; arbitrary partitions do not.

**Produces:** `reports/stability_results.csv` — mean and spread of agreement per k.

**Why it matters:** Arcan clustered once, with no error bars. This is the gap we fill, and it is the
single clearest justification for our final number.

**Time:** ~30 minutes.

---

### Step 6 — Naming the clusters ⬜

**Goal:** turn "cluster 7" into a category. This is the brief's *"subject categories for class
association."*

**What it does, three ways:**
1. **c-TF-IDF** — the words common in one cluster and rare in the others
2. **Representative abstracts** — the papers nearest each cluster centre, read by eye
3. **arXiv contingency table** — which arXiv subjects this cluster's papers actually carry, plus
   ARI/AMI against the reference taxonomy (**AMI**, not raw NMI, because NMI drifts upward with k)

**What we are looking for — the mismatches:**
- A cluster merging two arXiv categories → arXiv splits hairs the language doesn't support
- One arXiv category spread across clusters → arXiv is too coarse there
- A vague catch-all cluster → reported honestly; Arcan got one in every configuration

**Produces:** `reports/cluster_profiles.md` — one page per cluster, and the final category names.

**Decision it makes:** the category definitions, plus a sanity check on k. If clusters cannot be
named, k is wrong and we return to Step 5.

**Time:** ~20 minutes of compute, plus reading time.

---

### Step 7 — Train the classifier ⬜

**Goal:** a model that assigns one of our discovered categories to an unseen abstract. Instruction 4.

**What it does:** fine-tunes `bert-base-uncased` on (abstract → discovered category) pairs, using the
splits fixed in Step 3. Baselines: TF-IDF + LinearSVC, and a majority-class floor.

**How we report it:** because the targets come from our own clustering, the score measures
**agreement with the discovered taxonomy**, not accuracy against external truth. We state that
plainly, and additionally train one classifier on arXiv's real archive labels as a separate honest
reference point — a comparison Arcan never ran.

**Metric:** macro-F1, not accuracy, given the imbalance.

**Produces:** trained model, validation metrics, confusion matrix.

**Compute:** **must run on Google Colab (free T4).** On 16 CPU cores, bert-base over 150k samples is
roughly 60 hours. On a T4 it is 1.5–2 hours. Steps 3–6 stay local; this step exports the dataset and
runs in a Colab notebook committed to the repo.

---

### Step 8 — Tuning and model selection ⬜

**Goal:** instruction 6.

**What it does:** tunes learning rate, batch size and epochs (Arcan's ranges: LR 1e-6→1e-4, batch
8/16/32, 2–7 epochs), then compares `bert-base-uncased`, `distilbert-base-uncased` and `SciBERT`
across multiple seeds, so we report variation rather than one lucky run.

**Produces:** `reports/classification_results.csv`

**Compute:** Colab.

---

### Step 9 — Final report ⬜

**Goal:** instruction 7 — the deliverable.

**What it does:** fills README §2–7 with what we found, embeds the figures, documents the exact
commands to reproduce everything, and states limitations honestly — including where our evidence
does not settle a question.

**The headline result:** the number of categories, and the four kinds of evidence justifying it.

---

## 5. Decisions — all settled

Each was decided from measured evidence, not assumption.

| # | Decision | Settled as | Evidence |
|---|---|---|---|
| ① | Reference taxonomy | **All 20 archives, no cutoff** — and report against the **8 groups too** | 2016+ has only 20 archives; the ≥1% cutoff I first proposed would have deleted `econ` and `q-fin`, two whole disciplines. The year filter already removes the 24 legacy tags, so the cutoff was redundant. No single level is correct — 8 groups is too coarse for physics (13 fields → 1 class), 20 archives too coarse for `cs` (40 leaves → 1 class, 36% of papers) — so we report both |
| ② | Max input length | **256 tokens** | Measured: 128 tokens fits only **20.5%** of abstracts whole, 256 fits **67.2%**, 384 fits 96.0%. Arcan's 128 truncated four out of five abstracts |
| ③ | Sampling | **Stratified on primary archive, proportional.** Floor of ~100/archive **in the pilot only** | Guarantees every category is present rather than leaving it to luck. Balancing would make the discovered number of categories an artefact of our sampling. Imbalance is handled in the model (class weights + macro-F1), not in the data |
| ④ | Sample size | **Two phases:** pilot 5k cluster / 10k classify → main 50k cluster / 150k classify | Pilot matches Arcan exactly for direct comparison, runs on CPU, and catches bugs cheaply. Main size is driven by the rarest class: `nucl-ex` at 0.24% gives ~360 papers at 150k. If time runs out, the pilot alone is a complete submission |
| ⑤ | Year range | **2016 onwards** (2,118,113 papers) | Measured: physics falls 48%→36.5% and cs rises 26%→36.0%, so the corpus genuinely rebalances. All 24 legacy archives are pre-2016, so this removes them for free. Still 14× more papers than we need |
| ⑥ | Classifier compute | **Colab free T4 for phase 2 only**; everything else local | No local GPU (16 CPU cores). bert-base on 150k ≈ 60 h on CPU vs ~2 h on a T4. Phase-1 training (10k) is CPU-feasible |

### What the classifier predicts — to avoid any ambiguity

Iris confirmed interpretation **(b)**, so:

- The classifier's classes are **the categories we discover**, however many that turns out to be
- arXiv's 20 archives are **a measuring stick only** — used to validate clusters (ARI/AMI), to name
  them (step 6), and to stratify the sample. They are never the training target
- One *secondary* classifier is trained on arXiv's real archive labels purely as an honest reference
  point — the comparison Arcan never ran

---

## 6. Out of scope, and why

| Excluded | Reason |
|---|---|
| Fine-tuning the embedding model (contrastive / self-supervised) | No reviewed paper does it; days of GPU work; evidence suggests no gain |
| spaCy "triples" extraction | Arcan tested it and results got *worse* |
| Training the classifier on arXiv labels as the primary target | Explicitly rejected by Iris — that was interpretation (a) |
| Choosing k via ARI/NMI against true labels | Circular; the brief says the number is unknown |
| Full 3.2M corpus | Unnecessary; 150k is sufficient and CPU-feasible |
