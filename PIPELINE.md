# Pipeline walkthrough

What each phase does, why it is built that way, what it writes to disk, and what came
out of it. The README argues the result; this document traces the machinery.

Ten phases. Everything up to and including Phase 9 has been run end to end. Phase 10 is
specified and not run.

| # | Phase | Script | Ran on | Headline output |
|---|---|---|---|---|
| 0 | Download | `download_data.py` | 5.65 GB snapshot | 3,224,635 records |
| 1 | EDA | `eda_scan.py`, `eda_year_mix.py` | all records | 17.3% cross-group; mix drifts |
| 2 | Preprocess | `preprocess.py` | 2.1M eligible | 150,000 sampled, split 70/15/15 |
| 3 | Embed | `embed.py` | 150,000 | 150,000 × 384 matrix |
| 4 | Cluster | `cluster.py` | 150,000 | 23 scored configurations |
| 5 | **Choose k** | `stability.py` | 10k / 20k / 50k | **k = 8**, stable in all three sweeps |
| 6 | Interpret | `interpret.py` | 150,000 | 8 named categories |
| 7 | Baselines | `baseline.py` | 150,000 | tf-idf+SVM 0.9005 macro-F1 |
| 8 | Classifier | `train_classifier.py` | 7,022 (pilot) | BERT 0.9013 val / 0.9008 test |
| 9 | Verify & export | `check_integrity.py`, `figures.py`, `export_training_set.py` | both | all invariants hold |
| 10 | Tune | `tune.py` | — | **not run** (needs GPU) |

Two corpora run through phases 2–8: a **10,000-paper pilot** used to validate the
pipeline and replicate findings independently, and the **150,000-paper main corpus**
that produces every reported number. Where they disagree, the main corpus wins and the
disagreement is reported.

---

## Phase 0 — Acquire the data

**Script:** [`src/download_data.py`](src/download_data.py)

The Cornell arXiv metadata snapshot from Kaggle: one JSON Lines file, 5.65 GB, one
record per paper with id, title, abstract, category labels and version history.

Three details that matter:

**The version is pinned to 309.** Kaggle republishes this dataset monthly. Without a
pinned version the committed sample ids would drift against a moving corpus and the
results would stop being reproducible. `config.DATASET_VERSION` is the single place
this is set.

**The download falls back.** The newest version can return 404 while Kaggle is still
compressing it server-side, so the script tries `Cornell-University/arxiv` then
versions 308 and 307 in turn.

**The cache is forced outside the project folder.** This project lives inside a
OneDrive-synced directory. `KAGGLEHUB_CACHE` is pointed at `%LOCALAPPDATA%\kagglehub`
before `kagglehub` is imported, so 5.65 GB never syncs to corporate cloud storage. The
same rule governs every later artifact: derived data goes to
`%LOCALAPPDATA%\arxiv-iris\`, and only small committed results stay in the repo.

The token is read from the `KAGGLE_API_TOKEN` environment variable and appears in no
file in this repository.

**Output:** `%LOCALAPPDATA%\kagglehub\...\arxiv-metadata-oai-snapshot.json`
**Result:** 3,224,635 paper records.

---

## Phase 1 — Exploratory data analysis

**Scripts:** [`src/eda_scan.py`](src/eda_scan.py), [`src/eda_year_mix.py`](src/eda_year_mix.py)

Both stream the file line by line rather than loading it — at 5.65 GB a `read_json`
would not fit comfortably in memory, and nothing here needs random access.

`eda_scan.py` makes one pass collecting: abstract word-count distribution, labels per
paper, primary archive, primary group, papers per year, and the count of papers whose
labels span more than one top-level group. `eda_year_mix.py` makes a second pass
cross-tabulating year against archive and group.

Submission year is taken from `versions[0].created` — the *first* version — falling
back to `update_date`. Using `update_date` alone would date a 2003 paper revised in
2024 as 2024.

**Outputs:** `reports/eda_stats.json`, `reports/year_mix.json`, and three figures
(`eda_papers_per_year.png`, `eda_abstract_length.png`, `eda_archive_distribution.png`).

**Results, and what each one changed downstream:**

| Finding | Value | Consequence |
|---|---|---|
| Total records | 3,224,635 | — |
| Empty abstracts | 0 | no missing-text handling needed |
| Papers spanning >1 top-level group | 558,087 (17.3%) | the reference taxonomy is *soft*; recorded as a limitation in §7.5 |
| Papers with >1 subject label | ~half | same |
| Median abstract length | 159 words | 256 tokens fits 67.2% whole; 128 would fit only 20.5% → `MAX_TOKENS = 256` |
| Archive mix by year | drifts heavily | → the 2016 cutoff |

The year-mix result is the one that shaped the design most. arXiv's composition is not
stationary: computer science grows from a small share to the largest archive over the
corpus's lifetime. Sampling across all years would mean the discovered categories
partly reflect *when* papers were written. Cutting at 2016 removes all 24 legacy
archive codes and gives a corpus with a roughly stable field mix.

---

## Phase 2 — Preprocessing, sampling and splitting

**Script:** [`src/preprocess.py`](src/preprocess.py)

One streaming pass, applying two filters, then drawing a stratified sample per phase,
then splitting — in that order.

**Filters.** Submission year ≥ 2016; abstract ≥ 20 words (drops 15,567 stub
abstracts). 3,224,635 records → **2,113,230 eligible**.

**Text cleaning is deliberately minimal** — whitespace normalisation only. LaTeX is
left intact. Stripping it would delete real signal from mathematics abstracts, where
`\mathbb{R}` and `\mathcal{H}` are content words rather than noise, and the
sentence-transformer tokenisers handle them. This turns out to matter: `mathbb` and
`mathcal` are the two strongest c-TF-IDF terms for the pure-mathematics category
discovered in Phase 6.

**Sampling is stratified on the primary archive, preserving proportions.** Stratifying
guarantees every archive appears rather than leaving rare ones to chance.
*Not* rebalancing is the deliberate part: if the sample were flattened to equal
archive sizes, the number of categories discovered later would be an artefact of our
sampling rather than a property of the literature.

**The split is computed before the clustering subset is drawn**, and the clustering
subset comes only from the training portion. This ordering is what makes the
downstream classifier honest, and `check_integrity.py` verifies it in Phase 9.

**Outputs:**
- `%LOCALAPPDATA%\arxiv-iris\<phase>\sample.parquet` — the abstracts (large, regenerable)
- `data/splits/<phase>_ids.csv.gz` — ids + split + cluster flag, **committed to git**
- `reports/preprocess_summary.json`

Committing the ids is what makes the pinned dataset version useful: anyone can
reconstruct the exact 150,000 papers.

**Results:**

| | main | pilot |
|---|---|---|
| Sampled | **150,000** | 10,000 |
| train / val / test | 105,000 / 22,499 / 22,501 | 7,022 / 1,544 / 1,434 |
| Clustering subset (train-only) | 50,000 | 5,000 |
| Archives / groups present | 20 / 8 | 20 / 8 |
| Largest archive | cs — 53,930 (36%) | cs — 2,977 |
| Smallest archive | nucl-ex — 354 | nucl-ex — 119 |
| Multi-label | 52.7% | 54.0% |
| Median abstract | 159 words | 158 words |

The imbalance is severe — cs is 152× nucl-ex — which is why **macro-F1 is the headline
metric** everywhere downstream and accuracy is reported only to show how far a majority
guess gets without learning anything.

---

## Phase 3 — Embedding

**Script:** [`src/embed.py`](src/embed.py)

**The encoder is frozen. Nothing is trained here.** Every paper reviewed in §1.2 uses
off-the-shelf embeddings for scientific-document clustering; fine-tuning an encoder for
clustering is a separate research problem.

**Model:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, batch size 64,
truncated at 256 tokens.

The model choice comes from Arcan (2026), who benchmarked four candidates on this exact
task: all-mpnet-base-v2 (ARI 0.470), all-MiniLM-L6-v2 (0.442), allenai-specter (0.415),
scibert (0.304). Both science-specific models lost to the general-purpose pair. MiniLM
is ~5× faster than MPNet at close to the same quality, so it carries the reported
results; MPNet is implemented and configured but was not run at scale, which §7.5
records as a limitation.

**Only the abstract is encoded, with no title.** The brief specifies abstracts, and
Arcan found abstract-only input beat every richer variant they tried.

**Vectors are L2-normalised.** This makes Euclidean distance and cosine similarity
equivalent, which is what KMeans assumes — KMeans minimises Euclidean distance to
centroids, so feeding it unnormalised vectors would cluster partly on magnitude. It
also fixes the shape of the null model in Phase 5.

**Outputs:** `embeddings_minilm.npy` (float32, 150,000 × 384) plus a `.json` provenance
record. Cached; re-running skips unless `--force`.

---

## Phase 4 — Clustering sweep

**Script:** [`src/cluster.py`](src/cluster.py)

**Dimensionality reduction first.** UMAP to 50 dimensions, `n_neighbors=15`,
`min_dist=0.0`, `metric="cosine"`, seeded. Density estimates degrade badly in high
dimensions, and HDBSCAN needs the reduction to work at all. `min_dist=0.0` deliberately
produces tight, well-separated clumps — which helps density clustering and, as Phase 5
discovers, badly distorts any criterion measured inside that space.

**Two algorithms:**
- **KMeans**, k = 3…20, `n_init=10`, seeded — 18 configurations
- **HDBSCAN**, `min_cluster_size` grid, `metric="euclidean"` on the UMAP output — 5 configurations

Each configuration is scored on: silhouette (KMeans) or DBCV (HDBSCAN), noise
percentage, largest-cluster share, smallest-cluster size, and ARI/AMI against both
arXiv reference levels.

**The central methodological decision is here.** Arcan selected k by maximising
`0.5·ARI + 0.5·NMI` *against arXiv's true labels* — and conceded it "may inflate
reported ARI/NMI". That is circular when the brief states the number of categories is
unknown: it uses the answer to find the answer. In this pipeline **ARI and AMI are
computed and reported but never used to choose anything.** They are after-the-fact
validation and, in Phase 6, the device for *interpreting* what each cluster is.

To be precise about what does the choosing: **nothing in this phase selects k.**
`select_k()` in Phase 5 reads two columns — the stability mean and its standard
deviation — and no others. The internal measures below are not a weaker selection
criterion that stability supplements; they are reported because they are the textbook
approach and they *fail here*, which is what makes Phase 5 necessary rather than
exotic.

A note on what the firewall actually forbids. It is not "never look at arXiv's labels"
— the dataset ships with them and ignoring them entirely would make the taxonomy
uninterpretable and unvalidatable. It is that **no label-derived quantity may enter the
selection path**. That is a property of the code, and it is checkable: the selection
function's inputs are the stability curve alone.

AMI is reported alongside ARI because raw NMI rises mechanically with cluster count and
would reward splitting. Both reference levels are reported because neither is correct
alone: 8 groups treat 13 distinct physics fields as one class, while 20 archives treat
all of computer science as one class.

**Output:** `reports/clustering_main_minilm.csv` — 23 rows × 15 columns.

**Results — and why the internal scores were rejected:**

| Configuration | Score | Largest cluster | Verdict |
|---|---|---|---|
| KMeans k=4 | **silhouette 0.5516 — the optimum** | **45.6%** | degenerate |
| KMeans k=3 | silhouette 0.5024 | 45.3% | degenerate |
| KMeans k=8 | silhouette 0.4383 | 18.5% | balanced |
| HDBSCAN mcs=100 | **DBCV 0.631** | **82.7%**, 2 clusters | degenerate |
| HDBSCAN mcs=250 | DBCV 0.643 | 42.6%, 3 clusters | degenerate |

Both internal criteria reward degeneracy on this data. Silhouette's optimum puts nearly
half the corpus in one cluster; DBCV's best score sits on a two-cluster partition
holding 82.7% in one. Neither is a usable answer to "how many categories are there",
which is what motivates Phase 5.

---

## Phase 5 — Choosing the number of categories

**Script:** [`src/stability.py`](src/stability.py) — *this is the step that answers the brief*

**The question.** If the data genuinely contains k groups, clustering two different
samples of it should recover the same groups. If k is wrong, the algorithm has to
invent boundaries, and invented boundaries land somewhere different each time. This
needs no labels at all.

Method from Lange, Roth, Braun & Buhmann, *Stability-Based Validation of Clustering
Solutions*, Neural Computation 2004, 16(6):1299–1323; the subsample-pair estimator is
Ben-Hur, Elisseeff & Guyon, PSB 2002.

**The procedure, per candidate k:** draw two independent 80% subsamples, cluster each
separately, compute ARI between their labels *on the overlap only*, repeat 10 times,
report mean and standard deviation. Cost: 10 reps × 2 subsamples × 2 data sets
(observed and null) × 10 KMeans initialisations = 400 clusterings per k, 7,200 across
the sweep.

**Three design decisions, each of which was wrong first and had to be fixed.** All
three are recorded in README §7.4.

**(a) The null model.** ARI is chance-corrected, so random *labellings* score 0 — but
that is not the relevant null. The question is whether the data is more reproducible
than *structureless data of the same shape*, because KMeans will happily carve a
featureless blob the same way twice. The first attempt shuffled each embedding
coordinate independently; that preserves every marginal while destroying the joint
structure, which sounds right and is wrong here, because the coordinates are
multi-modal and the product distribution stays clusterable. It scored **0.9998** and
the resulting criterion selected k=16 on noise. Replaced with **points drawn uniformly
on the unit sphere** — genuinely structureless, and the correct shape because the real
vectors are L2-normalised. It scores 0.007.

**(b) The space.** Measured inside the UMAP reduction, stability saturates: 17 of 18
candidate values clear the threshold and the rule returns whatever the sweep's upper
limit happens to be. `min_dist=0.0` manufactures separated clumps, so what is being
measured there is UMAP's determinism, not structure in the data. **k is decided in the
original L2-normalised embedding space**, where 4 of 18 clear. `--space umap`
reproduces the saturated version, because the contrast is itself a reportable result.

**(c) The selection rule.** Ben-Hur et al. require scores that are both high *and*
concentrated, so the decision is made on a **one-standard-deviation lower bound**, not
the mean. On the pilot the mean alone selects k=11 (0.849 ± 0.067) over k=8
(0.942 ± 0.020), inverting the evidence. Among all k clearing the bar, the largest is
chosen: a two-way split is perfectly stable and useless, so the most granular
reproducible taxonomy wins.

**Outputs:** `reports/stability_<phase>_<model>_<space>_n<N>.{csv,json}` — one pair per
sweep, tagged with its sample size — plus `reports/stability_consensus.json` and
`reports/figures/stability_consensus_embedding.png`.

**The sweep was run three times**, on independent samples of 10,000, 20,000 and 50,000
abstracts. This was not redundancy for its own sake: the three curves disagree, and the
disagreement is the reason the selection rule is what it is.

| Sweep | k whose lower bound clears 0.80 |
|---|---|
| pilot, n=10,000 | 3, 4, 5, **8** — dips at 6 and 7 |
| main, n=20,000 | 3, 4, 5, 6, 7, **8** — collapses from 9 |
| main, n=50,000 | 3, 5, 6, 7, **8**, 9, 15, 16, 17, 18 — collapses at 10, revives at 15 |
| **stable in all three** | **3, 5, 8** |

**(d) The selection rule — two attempts discarded.** The original rule took the largest k
clearing the threshold anywhere in the sweep. On the 50,000-row run that returns **18**:
the curve collapses at k=10 as expected, then climbs back to a lower bound of 0.93 at
k=17, and a rule with no notion of a first stable block walks straight past the collapse.
That is the same defect this pipeline rejects in the UMAP space — a rule whose answer is
set by the sweep's upper limit. The obvious repair, stopping at the first sustained
collapse, returns **5** on the pilot, whose genuine dip at k=6–7 reads as a cliff.

What survives is replication. A number of categories is only meaningful if an independent
sample finds it again — the premise the whole method rests on, applied to the sweeps
themselves rather than only to the subsamples inside one sweep.

> **Rule:** the largest k whose one-standard-deviation lower bound clears 0.80 in
> **every** sweep. Strictly more conservative than any single-curve rule: it cannot
> select a k that any sweep found unstable, and it is indifferent to where a particular
> curve is ragged.

**Result:**

> **k = 8.** Lower bound 0.922 at n=10,000, 0.946 at n=20,000, 0.893 at n=50,000 —
> stable in all three, against a structureless null of 0.005–0.008. Only k=3, k=5 and
> k=8 clear everywhere; 8 is the largest, and among reproducible taxonomies the more
> informative is preferred.

No arXiv label was consulted at any point. **An independent check agrees:** HDBSCAN —
different algorithm, different space, given no k at all — settles on 7 on the pilot.

**What one sweep alone would have claimed.** Reported from the 20,000-row run only, this
phase would have described a textbook plateau from k=3 to k=8 with a clean cliff at k=9 —
real in that sample, absent in the other two. The number is unchanged; the evidence for it
is now agreement between samples rather than the shape of a single curve. Recorded in
README §7.4.

**Sample size caveat.** The estimator costs 7,200 clusterings per sweep, so each runs on a
capped subsample rather than all 150,000. This is the one result in the report not
computed on the full corpus. Sample size is part of every filename — an earlier build let
two sweeps differing only in `--max-n` overwrite each other, which briefly left one run's
curve stamped with another's metadata.

## Phase 6 — Interpreting the categories

**Script:** [`src/interpret.py`](src/interpret.py)

A number is not an answer to "report the best possible number of categories" — a
category has to mean something. Each cluster gets a profile from three independent
sources of evidence:

**1. c-TF-IDF distinguishing terms.** Grootendorst, *BERTopic*, arXiv 2022, eq. 1:
`W(t,c) = tf(t,c) · log(1 + A/f(t))`. Plain per-document TF-IDF would surface words
common to every abstract; treating each cluster as one aggregated document surfaces
what is specific to it.

**2. Representative abstracts** — the papers nearest each centroid, so the profile can
be sanity-checked by reading rather than trusted from term lists.

**3. arXiv composition** — which archives and groups land in each cluster. **This is
the only place the reference taxonomy is used**, and it is used to *interpret* a
cluster, never to choose or score one.

Homogeneity and completeness are reported separately rather than only as V-measure,
because they fail in opposite directions and the distinction is itself the finding: a
cluster set can be pure but fragmented (high homogeneity, low completeness) or coherent
but conflated.

**Outputs:** `labels_minilm_kmeans_k8_emb.npy` (reused by every downstream step),
`reports/cluster_profiles_main_minilm_kmeans_k8_emb.md`,
`reports/cluster_composition_main_minilm_kmeans_k8_emb.csv`,
`reports/cluster_interpretation_main_minilm_kmeans_k8_emb.json`, plus the archive
heatmap and UMAP scatter figures.

**Result — the eight discovered categories:**

| | Category | Papers | Share | Top terms | Dominant archive |
|---|---|---|---|---|---|
| c0 | Algorithms & optimisation | 28,859 | 19.2% | algorithm, problem, performance, systems | cs 51.4% |
| c1 | High-energy theory | 11,697 | 7.8% | theory, field, black, mass, matter | hep-ph 29.8% |
| c2 | **Language models** | 17,890 | 11.9% | models, language, **llms**, reasoning | cs **94.6%** |
| c3 | Differential equations | 16,063 | 10.7% | equation, solutions, numerical, boundary | math 59.2% |
| c4 | Pure mathematics | 17,026 | 11.4% | **mathbb**, **mathcal**, prove, group, algebra | math 85.9% |
| c5 | Astrophysics | 12,640 | 8.4% | mass, stars, galaxies, stellar | astro-ph 84.8% |
| c6 | Condensed matter | 20,346 | 13.6% | quantum, states, magnetic, phase, spin | cond-mat 49.6% |
| c7 | Machine learning & vision | 25,479 | 17.0% | learning, image, training, neural | cs 80.4% |

Every paper is assigned, no cluster is degenerate (largest 19.2%, smallest 7.8%), and
each is interpretable from its vocabulary alone.

**Agreement with arXiv** — reported, not optimised:

| Reference | Homogeneity | Completeness | V-measure |
|---|---|---|---|
| 20 archives | 0.469 | 0.490 | 0.479 |
| 8 groups | 0.526 | 0.357 | 0.425 |

**The finding that justifies the exercise is c2.** Large-language-model research —
11.9% of the corpus, 94.6% computer science, terms `llms` / `language` / `reasoning` —
has no arXiv archive, because the taxonomy predates the field. It is filed under
general `cs` beside operating systems and computational geometry. The method recovered
it twice, independently, at two scales fifteen times apart. A classifier trained on
arXiv's labels could not have found it, and no agreement score would have revealed it:
by construction, agreement with arXiv *penalises* discovering a category arXiv lacks.

---

## Phase 7 — Classical baselines

**Script:** [`src/baseline.py`](src/baseline.py)

A fine-tuned transformer is only worth its cost if it beats something cheap. These also
act as a lower bound on how learnable the discovered taxonomy is: if a linear model on
bag-of-words already separates the clusters, the categories are at least lexically
coherent.

Three baselines, in increasing order of what they know:

- **majority** — always predicts the largest cluster. Exists to show how misleading
  accuracy is under imbalance.
- **tf-idf + LinearSVC** — word 1–2 grams plus character 3–5 grams (`char_wb`),
  `class_weight="balanced"`. The standard strong baseline for topical text
  classification.
- **embeddings + logistic regression** — **diagnostic only**. The clusters were
  *defined* in this space, and a KMeans partition is a Voronoi tessellation of it, so
  the task is linearly separable here by construction. The number measures how cleanly
  the clustering cut the space, not whether the categories generalise from raw text.

**Output:** `reports/baselines_main_minilm_kmeans_k8_emb.json`

**Result (150,000 papers, 8 discovered categories):**

| Baseline | val macro-F1 | val accuracy | test macro-F1 | Fit |
|---|---|---|---|---|
| majority | 0.0404 | 0.1929 | — | 0s |
| tf-idf + LinearSVC | **0.9005** | 0.8971 | 0.8970 | 355s |
| embeddings + logreg *(diagnostic)* | 0.9808 | 0.9805 | — | 192s |

Per-category F1 for the SVM runs 0.86–0.94 with no weak class. Note the gap between
majority macro-F1 (0.0404) and majority accuracy (0.1929) — that ratio is the argument
for macro-F1 in one line.

---

## Phase 8 — Transformer classifier

**Script:** [`src/train_classifier.py`](src/train_classifier.py)

**Configuration:** `bert-base-uncased`, lr 2e-5, batch 16, 3 epochs, 256 tokens, weight
decay 0.01, 131 warmup steps, class-weighted cross-entropy, early stopping patience 2,
seed 42. Device: CPU.

**Class weights go in the loss, not into resampling.** Inverse-frequency weights,
mean-normalised, passed through Trainer's `compute_loss_func` hook. Resampling would
change the effective corpus; weighting changes only the gradient.

**The best epoch is chosen on validation macro-F1.** The test split is scored once and
reported, never consulted during training.

**Scope — this is the honest limitation of the phase.** It ran on the **pilot**
(7,022 training abstracts), not the main corpus, and on **one configuration with one
seed**. A measured CPU run projects to ~52 hours for a single full-scale run and roughly
three weeks for the ten-run search, which is why Phase 10 needs a GPU. This run
exists so the report carries a real transformer number rather than a projection.

**Outputs:** `reports/classifier_pilot_bertbase_k8_cpu.json` (hyperparameters,
per-class scores, confusion matrix, library versions, wall-clock),
`reports/figures/confusion_matrix_pilot.png`

**Result:** 349.5 minutes. Validation macro-F1 by epoch: 0.8539 → 0.8954 → **0.9013**.

| Split | macro-F1 | weighted F1 | accuracy |
|---|---|---|---|
| validation | 0.9013 | 0.8994 | 0.8996 |
| test | 0.9008 | 0.8976 | 0.8975 |

Per-class test F1 runs 0.86 (c1) to 0.96 (c3) — **no category collapses**, and macro and
weighted scores sit within 0.004 of each other, so the class weighting worked.

**Where the errors are.** The three computer-science-derived categories trade mistakes
among themselves: 8% of language-model papers (c2) and 8% of deep-learning papers (c7)
are predicted as algorithms/optimisation (c0), and 6% of c2 goes to c7. Astrophysics is
cleanest at 97%, and the physics and mathematics categories are nearly disjoint from the
rest. This mirrors the clustering exactly: the cs/ML region of the embedding space is
one dense mass that k=8 cuts into three, and the classifier inherits those boundaries.

*(Cluster numbering in this phase is the pilot's, which differs from the main corpus's
in Phase 6.)*

---

## Phase 9 — Verification, figures and export

**Scripts:** [`src/check_integrity.py`](src/check_integrity.py),
[`src/figures.py`](src/figures.py),
[`src/export_training_set.py`](src/export_training_set.py)

**Integrity checks.** Every later step joins embeddings, cluster labels and abstracts
*by row position*. A single misaligned file would corrupt every number downstream
without raising an error anywhere, so the invariants are checked explicitly and the
exit code is non-zero on failure:

- sample sizes and split fractions match the configuration
- both filters (year ≥ 2016, ≥ 20 words) hold on every retained row
- all 20 archives appear in every split
- **the clustering subset contains zero validation or test rows** — verified at 0 for
  both phases, which is the no-leakage claim made concrete
- committed ids match the sample in order
- embeddings are L2-normalised and row-aligned

*One trap this caught:* pandas parses arXiv ids as float64, so the id check failed
spuriously. All 160,000 ids were verified to round-trip exactly (the 2016 cutoff
removes old-style ids), and the fix is `dtype={"id": str}`, documented in
`preprocess.py`.

**Figures.** 18 PNGs. The categorical palette was validated for colour-vision
deficiency across all adjacent pairs; the eight-colour UMAP scatter failed all-pairs
validation, so it is rendered as small multiples plus a directly-labelled overview
rather than one eight-colour plot.

**Export.** `export_training_set.py` bundles exactly what GPU training needs — abstract,
split, discovered label — into one self-contained file with a manifest recording its
sha256, class counts and split sizes. Everything that *defines* the experiment stays
here, seeded and committed, so the notebook **cannot re-derive the labels differently**.
Noise rows are kept as `y = -1` rather than dropped, so the consumer chooses how to
handle them.

**Output:** `trainset_main_minilm_kmeans_k8_emb.parquet` — 94.6 MB,
sha256 `71568b1c134fef65`, 150,000 abstracts, 8 categories, 0 noise.

---

## Phase 10 — Hyperparameter search — **not run**

**Script:** [`src/tune.py`](src/tune.py) (specified, `--dry-run` prints the exact commands)
**Notebook:** [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) (21 cells, runs the same search on a T4)

A full grid over learning rate × batch size × encoder × seed is 4 × 3 × 3 × 3 = **108
runs**, and almost all of it re-measures what the first few cells established. Instead
each factor is resolved in turn and the winner carried forward — **10 distinct runs** for the
same decisions:

| Stage | Varies | Values | Runs |
|---|---|---|---|
| 1 | learning rate | 1e-5, 2e-5, 3e-5, 5e-5 | 4 |
| 2 | batch size | 8, 16, 32 | 2 |
| 3 | encoder | bert-base, DistilBERT, SciBERT | 2 |
| 4 | seeds | 42, 1337, 2024 | 2 |

Learning rate goes first because it dominates fine-tuning outcomes. SciBERT is included
despite Arcan finding its *embeddings* worst of four for clustering — embedding quality
and fine-tuning quality are different properties, and the comparison costs one run.
Stage 4 is not tuning: it re-runs the winner under three seeds so the headline number
carries a spread. Arcan reported no error bars at all; a single fine-tuning run is not
a measurement.

**The full grid is also implemented.** `tune.py --search grid` runs the complete
4 x 3 x 3 x 3 Cartesian product — 108 fine-tuning runs — making no separability
assumption, so learning-rate x batch-size interactions are measured rather than assumed
away. It shares the staged search's runner, selection rule and outputs. Neither has been
run; at roughly 52 CPU-hours per run at full corpus scale the grid is a cluster workload.

**Hyperparameters used in the one run that happened** (Phase 8): lr 2e-5, batch 16, 3
epochs, 256 tokens, AdamW with weight decay 0.01 — the defaults recommended in Devlin et
al. (2019, Appendix A.3), and within the Optuna search space Arcan used. Arcan reports
that space but not the values it selected, so the configurations cannot be matched
exactly. We differ deliberately on sequence length: 256 tokens against their 128, because
256 fits 67.2% of abstracts whole and 128 fits 20.5% (Phase 1).

**To run it:** upload the Phase 9 parquet to Colab, open the notebook, commit the two
result files it writes.

---

## What is settled and what is not

**Settled, at full scale, reproducible from this repo:**
the corpus and splits; the embedding; the clustering sweep; the number of categories and
the criterion that chose it; the eight categories and their interpretation; the
agreement scores; the classical baselines; every integrity invariant.

**Measured but narrow:** the transformer. 0.9013 / 0.9008 macro-F1 is real, but from one
configuration, one seed, at pilot scale.

**Not run:** the hyperparameter search (Phase 10), MPNet embeddings at scale, and the
50,000-paper train-only clustering robustness check.

**Known soft spots, all recorded in README §7.5:** k=8 is the most granular *stable*
option rather than a unique truth — k=3 through k=8 are all reproducible, and a
four-category taxonomy would be a defensible coarser answer; the reference taxonomy is
itself soft, since over half of papers carry multiple labels; KMeans was fitted on all
150,000 papers, so validation and test abstracts contributed to defining the categories
the classifier is scored on; and HDBSCAN's sweep used 50,000 papers because the full set
exhausted memory.
