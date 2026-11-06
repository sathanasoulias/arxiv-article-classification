# Research Article Classification on arXiv

**Discovering how many categories exist in a corpus of scientific abstracts, and training a
classifier to predict them.**

Data: the public [arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv)
from Cornell University — 3,224,635 records, 5.65 GB, version 309.

---

### Summary

The number of categories in this corpus is unknown, so it has to be discovered rather than
assumed. We encode 150,000 abstracts with a frozen sentence-transformer, cluster them, and
choose the number of categories by **reproducibility**: a grouping counts only if an
independent sample of the same corpus recovers it. Measured across three independent sweeps
and against a structureless null, **eight categories** survive as the most granular
reproducible taxonomy. No arXiv label is consulted in reaching that number.

The resulting taxonomy is structurally the inverse of arXiv's — it splits computer science
into three and mathematics into two while covering thirteen physics archives with three —
and it contains one category arXiv has no archive for: **large-language-model research,
11.9% of the corpus**. A classifier trained on arXiv's own labels could not have found it.

The discovered categories are learnable from raw text: TF-IDF with a linear SVM reaches
**0.9005 macro-F1** on 150,000 abstracts, and a fine-tuned `bert-base` reaches **0.9013** at
pilot scale. The hyperparameter search is implemented and was not run; §6.3 says so.

### Contents

| | Section |
|---|---|
| **1** | [Problem, background and approach](#1-problem-background-and-approach) |
| **2** | [Exploratory data analysis](#2-exploratory-data-analysis) |
| **3** | [Preprocessing](#3-preprocessing) |
| **4** | [Model training](#4-model-training) |
| **5** | [Evaluation](#5-evaluation) |
| **6** | [Hyperparameter tuning and model selection](#6-hyperparameter-tuning-and-model-selection) |
| **7** | [Conclusions](#7-conclusions) |

Two companion documents: [PIPELINE.md](PIPELINE.md) traces every phase with its parameters
and outputs; [PRESENTATION.md](PRESENTATION.md) is slide-by-slide speaking notes.

---

# 1. Problem, background and approach

## 1.1 Why this is not ordinary text classification

In ordinary text classification you are handed a list of categories and you train a model to
pick one. Here **the number of categories is unknown, and determining it is part of the
task.** That single difference reshapes everything downstream.

It makes this two problems rather than one:

- **An open question** — how many categories are there, and what are they?
- **A supervised question** — can a model read an unseen abstract and assign it correctly?

The arXiv data does ship with subject labels such as `cs.LG` or `math.PR`, and the obvious
shortcut is to train against them. We do not, for three reasons.

**Authors choose their own labels**, so they are applied inconsistently across a corpus
spanning three decades and thousands of submitters.

**Most papers carry several labels at once.** In this snapshot **52.7%** of sampled papers
have more than one subject label and **17.3%** span more than one top-level group, so "the"
category of a paper frequently does not exist. Any hard partition is being compared against
a reference that is itself soft.

**The taxonomy is older than parts of the field it describes.** arXiv has no archive for
large-language-model research; that work is filed under general `cs` beside operating
systems and computational geometry. A method anchored to arXiv's labels is structurally
incapable of discovering a category arXiv lacks — and §4.4 shows this corpus contains one.

So the work splits into four questions, each tested separately:

| | Question | Why it is separable |
|---|---|---|
| 1 | **Representation** — how is an abstract turned into numbers? | Encoder choice is independent of what you do with the vectors |
| 2 | **Discovery** — what natural groups exist? | Algorithm choice is independent of how many groups you keep |
| 3 | **Cardinality** — how many groups, and how is that defended? | The assignment's central question |
| 4 | **Classification** — can a model label unseen abstracts? | Measured on a held-out split, within one label set |

Treating them separately means each decision is tested on its own evidence, rather than
bundled into one pipeline that either works or does not.

## 1.2 Background

This section covers the methods the project rests on. It is deliberately general; the
specific prior study closest to our task is discussed separately in §1.3.

### Representing text as vectors

Three different numeric representations appear in this project, doing three different jobs.
It is worth separating them up front, because only one of them feeds the clustering:

| Representation | Where | Used for |
|---|---|---|
| **Sentence encoder** — `all-MiniLM-L6-v2`, frozen, 384d, L2-normalised | `embed.py` | **The clustering.** This is how abstracts become numbers for discovery. |
| **TF-IDF** — word 1–2 grams + char 3–5 grams | `baseline.py` | A baseline *classifier* only. Never an input to the clustering. |
| **c-TF-IDF** — cluster-level term weights | `interpret.py` | *Naming* clusters once they exist. Not a document representation. |
| **Fine-tuned BERT** | `train_classifier.py` | Learns its own representation end to end for classification. |

The classical representation is **TF-IDF** over a bag of words — cheap, strong on topical
tasks, and blind to word order and synonymy. It remains a serious baseline, and we use it as
one (§5.2).

**BERT** (Devlin et al., 2019, [arXiv:1810.04805](https://arxiv.org/abs/1810.04805))
replaced this for most tasks: a transformer pretrained with masked-language modelling,
producing contextual representations that are fine-tuned per task. But BERT has a known
weakness for *comparing* texts. Its `[CLS]` vector was never trained to place similar
documents near each other, and using it directly as a sentence embedding performs poorly on
similarity tasks.

**Sentence-BERT** (Reimers & Gurevych, 2019,
[arXiv:1908.10084](https://arxiv.org/abs/1908.10084)) addresses this directly, training BERT
in a siamese configuration so that cosine distance between outputs is meaningful. This is
the family we encode with, because clustering is entirely a question of distances.

Science-specific encoders exist. **SciBERT** (Beltagy et al., 2019,
[arXiv:1903.10676](https://arxiv.org/abs/1903.10676)) pretrains on a scientific corpus with
its own vocabulary; **SPECTER** (Cohan et al., 2020,
[arXiv:2004.07180](https://arxiv.org/abs/2004.07180)) trains on the citation graph so that
co-cited papers land close together. **SciRepEval** (Singh et al., 2023,
[arXiv:2211.13308](https://arxiv.org/abs/2211.13308)) then evaluated such embeddings across
24 tasks and found no single representation wins everywhere — the best choice depends on the
task.

> **What we take from this.** Use an encoder trained for similarity rather than raw BERT,
> and do not assume the domain-specific model is better. Test it.

### Grouping without labels

Clustering algorithms divide along a line that matters a great deal here: whether the number
of groups is an input or an output.

**Partitional methods take k as input.** **KMeans** minimises within-cluster variance and
assigns every point, producing a Voronoi tessellation of the space — which implies convex,
roughly isotropic clusters of comparable size. **Gaussian mixture models** relax this to
soft, elliptical assignments. Both require k to be decided elsewhere, which is precisely the
thing we do not know.

**Density methods infer the number themselves.** **DBSCAN** (Ester et al., 1996) grows
clusters from dense regions and labels sparse points as noise, finding arbitrarily shaped
clusters without being told how many. **HDBSCAN** (Campello et al., 2013) extends it across
density scales, removing DBSCAN's single global `eps`. The appeal here is obvious: an
algorithm that answers the assignment's central question by itself.

The catch is that density estimation degrades badly in high dimensions — distances
concentrate and density becomes nearly uniform. The standard remedy is to reduce first,
usually with **UMAP** (McInnes et al., 2018,
[arXiv:1802.03426](https://arxiv.org/abs/1802.03426)), which preserves local neighbourhood
structure better than PCA for this purpose.

**BERTopic** (Grootendorst, 2022, [arXiv:2203.05794](https://arxiv.org/abs/2203.05794))
packages exactly that sequence — embed, UMAP, HDBSCAN — and then describes each cluster with
**class-based TF-IDF**, which aggregates a cluster into a single pseudo-document so the
terms that distinguish it surface instead of the terms common to every abstract. It is the
modern successor to **LDA** (Blei et al., 2003) for topic discovery, and we use its c-TF-IDF
formulation to make clusters readable (§4.4).

> **What we take from this.** UMAP before density clustering, and c-TF-IDF for
> interpretation. But published BERTopic results also carry a warning: reported topic counts
> vary enormously on the same data, with cases producing thousands of topics where humans
> labelled a handful, and a large share of documents dumped into the noise bin. We therefore
> never report an HDBSCAN cluster count as *the* answer, and always report the noise
> fraction beside it.

### Choosing the number of groups

This is the hard part, and the literature offers two families.

**Internal validity indices** score a partition on its geometry. The **silhouette**
coefficient (Rousseeuw, 1987) compares within-cluster distance to nearest-other-cluster
distance; **DBCV** (Moulavi et al., 2014) is its density-based analogue, built for the
non-convex clusters where silhouette is invalid. The usual recipe is to sweep k and keep the
optimum.

Their weakness is that they measure *shape*, not *correctness*. In a continuous embedding
space the most compact partition is often a degenerate one, and §4.2 shows exactly that
happening here — both indices peak on solutions where a single cluster holds most of the
corpus.

**Stability selection** asks a different question: *is the grouping reproducible?* If the
data genuinely contains k groups, clustering two independent samples should recover the same
groups; if k is wrong, the algorithm must invent boundaries, and invented boundaries fall
differently each time. The approach originates with **Ben-Hur, Elisseeff & Guyon (2002)**,
who introduced the subsample-pair estimator, and **Lange, Roth, Braun & Buhmann (2004)**,
*Stability-Based Validation of Clustering Solutions*, Neural Computation 16(6):1299–1323.
**von Luxburg (2010)** surveys the method, including where it fails.

Crucially, stability needs **no labels at all** — which is what makes it usable when the
number of categories is unknown by definition.

> **What we take from this.** Stability is our selection criterion. Internal indices are
> computed and reported, but as evidence of *why* they cannot be used here, not as a
> contribution to the decision.

### Comparing two groupings

Both stability selection and after-the-fact validation need a way to compare partitions. The
**Rand index** (Rand, 1971) counts the pairs of items two partitions agree about; the
**Adjusted Rand Index** (Hubert & Arabie, 1985) corrects it for chance so that random
agreement scores zero. **Normalised mutual information** is the information-theoretic
equivalent, and **AMI** its chance-corrected form (Vinh, Epps & Bailey, JMLR 2010).

The distinction matters here: **raw NMI drifts upward as the number of clusters grows**, so
using it while choosing k would quietly reward splitting. We use ARI and AMI throughout.

### Evaluating an imbalanced classifier

Our categories differ in size by more than a factor of two, and the underlying archives by a
factor of 152. **Accuracy is the wrong headline** under that imbalance, being dominated by
the largest classes. We report **macro-F1**, which weights every category equally. §5.2
makes the case with our own measurements: a majority-class baseline reaches 19.3% accuracy
and 0.0404 macro-F1 on the same data.

### The three papers closest to our implementation

The brief asks for three articles referencing the implementation used. These are the three
whose methods appear directly in our code, rather than the three most similar in topic:

| # | Paper | What we implement from it |
|---|---|---|
| 1 | **Reimers & Gurevych (2019)** — Sentence-BERT | The representation: a similarity-trained encoder, `all-MiniLM-L6-v2`, used frozen (§4.1) |
| 2 | **Grootendorst (2022)** — BERTopic | The discovery recipe — UMAP before density clustering — and c-TF-IDF for naming clusters (§4.2, §4.4) |
| 3 | **Lange, Roth, Braun & Buhmann (2004)** | The selection criterion: stability across resamples, which answers the brief's central question (§4.3) |

Supporting references: Ben-Hur et al. (2002) for the subsample-pair estimator we actually
compute; Hubert & Arabie (1985) and Vinh et al. (2010) for the agreement measures; Devlin et
al. (2019) for the classifier; McInnes et al. (2018) for the reduction; Campello et al.
(2013) for HDBSCAN.

### What reading all this changed

The literature disagrees with itself on every one of our three open choices — which encoder,
which clustering algorithm, how to pick k — and the study closest to our task found that the
*simpler* option won on two of them. So we do not commit to one encoder, one algorithm or
one number up front. We write down testable predictions (§1.6) and run a controlled
comparison, then score the predictions against what happened (§7.3).

## 1.3 Related work on this dataset

**Scharpf et al. (2020), [arXiv:2005.11021](https://arxiv.org/abs/2005.11021)** (JCDL)
classified and clustered arXiv content with pre-transformer methods — doc2vec and TF-IDF on
4,900 documents — reaching 82.8% classification accuracy and 69.4% cluster purity. It is
useful as problem history and as a floor, not as a method we implement.

**Arcan (2026), [arXiv:2601.08841](https://arxiv.org/abs/2601.08841)** — *Triples and
Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents* — is
the closest published study: it clusters arXiv papers and trains classifiers on them,
comparing four encoders (MiniLM, MPNet, SciBERT, SPECTER) and three clustering algorithms.
Two of its findings informed our setup, and we adopt them as hypotheses rather than as
facts: plain abstract text outperformed richer representations that added structured
knowledge, and the two general-purpose sentence encoders beat the two science-specific ones
at clustering.

**It is not, however, the same task.** Arcan selects the number of clusters by maximising
`0.5·ARI + 0.5·NMI` **against arXiv's true labels**, and concedes the choice "may inflate
reported ARI/NMI". That is a legitimate design when the labels are taken as ground truth. It
is circular under this brief, where the number of categories is stated to be unknown: it
uses the answer to find the answer, and it structurally cannot discover a category arXiv
does not already have.

The practical consequences are visible in their results. The chosen cluster count moves with
the input representation — 6, 8, 10 and 8 across four variants — and no run carries error
bars. Their classification figure of roughly 0.923 macro-F1 is measured on a filtered subset
against a different label set, so it is a reference point and **not** a target: a macro-F1
over six categories cannot be compared with one over eight.

> **What we take from this.** The encoder shortlist and the abstract-only input, both as
> testable choices. What we deliberately do not take is the selection criterion.

## 1.4 What we were asked to predict

The brief admits two readings, so we asked rather than guessed. Iris.ai confirmed the
second:

> "You're right, (b) is the better interpretation. Discovering meaningful categories is an
> important part of the objective, but we leave the exact methodology up to you."

The task is therefore settled:

- Categories are **discovered from the abstracts**. Their number is an output, not an input.
- The classifier predicts **those discovered categories**, not arXiv's subject labels.
- arXiv's labels are used **only** to interpret and sanity-check. They never influence how
  many categories we report.

### Three questions, three separate instruments

The central methodological risk is answering two different questions with one number. We
keep them apart:

| Question | Instrument | What it may **not** be used for |
|---|---|---|
| How many categories are there? | Stability across independent resamples | Never uses arXiv labels |
| What are those categories about? | c-TF-IDF terms, nearest-centroid abstracts, arXiv composition | Not a score |
| Can a model assign a new abstract? | Macro-F1 on a held-out split | Only comparable within one label set |

In particular, **a classifier's macro-F1 is not evidence that a taxonomy is good.** Two
classifiers trained on different label sets cannot be ranked against each other — a trivial
two-category split would score near-perfectly and mean nothing. Taxonomy quality and
classifier quality are measured by different instruments, and we never trade one against the
other. §5.3 returns to this.

## 1.5 The pipeline

![Pipeline](reports/figures/pipeline.png)

The diagram is the argument in one picture. Read it for two things: the **single green box**
is the only representation the clustering ever sees, and the **dashed red box** is the only
place arXiv's own labels are allowed to enter — pointing at interpretation, after the number
has already been decided.

### What the data actually is at each step

| Step | Object | Concretely |
|---|---|---|
| Snapshot | JSON Lines | 3,224,635 records, 5.65 GB |
| After filters | — | 2,113,230 eligible abstracts |
| Sample | `sample.parquet` | 150,000 rows: id, abstract, archive, group, split |
| Embeddings | `embeddings_minilm.npy` | **150,000 × 384** float32, every row unit-length |
| UMAP | `umap_minilm_50d.npy` | 150,000 × 50 float32 |
| Cluster sweep | `clustering_main_minilm.csv` | 23 rows × 15 columns, one row per configuration |
| Stability | 3 × `stability_*_n*.csv` | 18 rows each (k=3..20), mean and sd over 10 repeats |
| Labels | `labels_minilm_kmeans_k8_emb.npy` | 150,000 integers in 0..7 |
| Training bundle | `trainset_*.parquet` | 94.6 MB, sha256 `71568b1c134fef65` |

Ten stages, each a script that caches its output so later stages can re-run without redoing
the work. [PIPELINE.md](PIPELINE.md) documents each one in full.

| # | Stage | Script | Output |
|---|---|---|---|
| 1 | Scan the snapshot | `eda_scan.py`, `eda_year_mix.py` | Corpus statistics, year × archive mix |
| 2 | Sample and split | `preprocess.py` | 150,000 abstracts, fixed 70/15/15 split |
| 3 | Embed | `embed.py` | Frozen sentence embeddings, L2-normalised |
| 4 | Reduce and sweep | `cluster.py` | UMAP to 50d; KMeans k=3..20 and HDBSCAN scored |
| 5 | **Choose the number** | `stability.py` | Stability curves against a structureless null |
| 6 | Interpret | `interpret.py` | Per-category terms, examples, arXiv composition |
| 7 | Baselines | `baseline.py` | Majority, TF-IDF+SVM, embeddings+logreg |
| 8 | Classify | `train_classifier.py` | Fine-tuned transformer |
| 9 | Verify and export | `check_integrity.py`, `figures.py`, `export_training_set.py` | Invariants, figures, GPU bundle |
| 10 | Tune | `tune.py`, `notebooks/colab_train.ipynb` | Staged search (10 runs) or full grid (108) |

Four decisions inside that pipeline are worth justifying up front, because in each case the
obvious choice is the wrong one.

**The encoder is frozen; nothing is trained to produce embeddings.** Every study reviewed in
§1.2 uses off-the-shelf sentence embeddings for scientific-document clustering. Fine-tuning
an encoder *for clustering* is a separate research problem, and the prior evidence suggests
the specialised models would not help.

**Only the abstract is encoded — no title.** The brief specifies abstracts, and the closest
prior study found abstract-only input beat every richer variant it tried.

**UMAP reduction comes before density clustering**, because density estimates degrade in
high dimensions. Arcan ran HDBSCAN on raw embeddings and it collapsed to 3–5 clusters with
up to 74% of papers marked noise, concluding HDBSCAN was the worst of three algorithms. We
treat that as a fixable artefact rather than a property of the algorithm, and §4.2 reports
what happens when it is fixed.

**But the number of categories is *not* decided in UMAP space.** UMAP with `min_dist=0.0`
deliberately manufactures tightly separated clumps, and every internal criterion saturates
inside that space — measuring UMAP's determinism rather than structure in the data. The
reduction is used where it is needed, and the number of categories is settled in the
original embedding space (§4.3).

### Keeping the scope realistic

The full snapshot is 3.2 million papers and the full hyperparameter grid is 108 training
runs. Neither is necessary, and both were cut deliberately:

- One fixed stratified sample of **150,000** abstracts, drawn once and committed as arXiv IDs
- A **10,000**-paper pilot to validate every step cheaply, which doubles as an independent
  replication at a fifteenth of the scale
- Embeddings computed once and cached
- A staged hyperparameter search costing **10** distinct runs instead of 108 (§6.1)

## 1.6 Predictions

Written in advance as statements we can be wrong about, and scored honestly in §7.3.

- **H1** — Silhouette and stability will disagree about the number of categories. Where they
  do, we prefer stability and say so.
- **H2** — HDBSCAN *after* UMAP reduction will not show the collapse Arcan reported, and
  will leave far less than 74% of papers as noise.
- **H3** — The discovered categories will not line up with arXiv's 8 top-level groups. We
  expect them to merge several non-physics groups and split physics, because arXiv's
  taxonomy reflects community history while abstracts cluster by shared method and
  vocabulary.
- **H4** — The two reference levels will peak at different numbers of clusters: the 8 groups
  favouring fewer categories, the 20 archives more.
- **H5** — A fine-tuned BERT will beat a TF-IDF + LinearSVC baseline on macro-F1. If it does
  not, that is a finding about how lexically separable the discovered categories are, not a
  bug.

## 1.7 What this study contributes

A reproducible comparison of ways to represent and group scientific abstracts, reporting a
number of categories **together with the criterion that justifies it**, showing how those
categories relate to arXiv's existing taxonomy, and measuring how well a transformer
classifier generalises to unseen abstracts.

It also documents its own limits. Category granularity is genuinely ambiguous, scientific
fields overlap, and discovered clusters are not the same kind of object as human-assigned
subject labels. Where the evidence does not settle a question, we say so rather than
reporting the flattering number.

> **Reading this report.** Sections 1.2–1.6 were written before the experiments and state
> what the literature suggested and what we expected. Everything from §2 onward reports what
> happened. §7.3 scores the predictions against the results, and §7.4 records the four
> places our own method was wrong and had to be fixed — including, twice, a conclusion we
> had already written down.

---

## 2. Exploratory data analysis

`src/eda_scan.py` streams the 5.65 GB snapshot once and never loads it into memory.
`src/eda_year_mix.py` adds the year × subject cross-tabs. Both write JSON to `reports/`.

**Snapshot v309 — 3,224,635 papers.**

| Property | Value | Why it matters |
|---|---|---|
| Papers | 3,224,635 | Pinned to Kaggle version 309 |
| Empty abstracts | 0 | Nothing to impute |
| Abstracts under 20 words | 15,567 | Dropped — too short to carry topical signal |
| Median abstract length | 145 words | 256 wordpiece tokens fits 67.2% whole; 128 would fit 20.5% |
| Papers with >1 subject label | 48.3% | **There is no single "true" category per paper** |
| Mean labels per paper | 1.74 (max 13) | |
| Papers spanning >1 top-level group | 558,087 (17.3%) | Overlap is not just within a field |
| Distinct archives | 38 (20 still active from 2016) | 18 are legacy archives, retired before 2016 |

![Papers per year](reports/figures/eda_papers_per_year.png)

![Abstract length](reports/figures/eda_abstract_length.png)

Three findings shaped the rest of the work.

**Nearly half of all papers carry more than one label.** This is the single most important fact in
the dataset. It means arXiv's labels cannot serve as ground truth for a single-label task — there
is no one right answer to compare against. It is a large part of why we treat the arXiv taxonomy as
a measuring stick for interpretation rather than as a target.

**The subject mix shifts sharply over time.** Computer science was a minor archive in the 1990s and
is now the largest. A sample drawn across all years would be a sample of arXiv's *history*, and the
categories we discovered would partly reflect how the archive's composition changed rather than how
research topics differ. We therefore restrict to **2016 onward** (2,118,113 papers), which also
removes all 18 legacy archives cleanly and leaves 20 active ones.

![Archive distribution](reports/figures/eda_archive_distribution.png)

**Class imbalance is extreme and we keep it.** In the drawn sample the largest archive (`cs`) has
53,930 papers and the smallest (`nucl-ex`) has 354 — a ratio of roughly 150:1. Rebalancing would
make the number of discovered categories an artefact of our sampling, so the real proportions are
preserved and the imbalance is handled later, in the classifier's loss function.

### A decision worth recording: no minimum-share cutoff

An earlier version of this work kept only the 14 archives above a 1% share, which covered 96.5% of
papers. Recomputing on 2016+ data showed the cutoff would keep only 12 archives and **delete `econ`
and `q-fin` entirely**. Since one of the questions is whether the discovered categories match
arXiv's, silently removing two of them would have prejudged the answer. All 20 active archives are
sampled, with no cutoff.

## 3. Preprocessing

`src/preprocess.py` makes one streaming pass over the 5.65 GB snapshot and writes a fixed
sample, a fixed split, and a committed list of arXiv IDs. Three decisions are made here, and
each is one a reader should be able to check rather than take on trust.

![Sampling](reports/figures/sampling_funnel.png)

### What the filters remove, and why

| Filter | Records | Rationale |
|---|---|---|
| arXiv snapshot v309 | 3,224,635 | — |
| Submitted 2016 or later | 2,118,113 | removes all 24 legacy archive codes and the era when the corpus was overwhelmingly physics |
| Abstract of at least 20 words | **2,113,230** | drops 4,883 stub abstracts that carry no topical signal |
| Stratified sample drawn | **150,000** | 7.1% of the eligible frame; see below |

**The 2016 cutoff is the consequential one.** arXiv's composition is not stationary —
computer science grows from a minor archive to the largest one over the corpus's lifetime
(§2). Sampling across all years would mean the discovered categories partly encode *when*
papers were written rather than what they are about. The cost is stated plainly in §7.5: the
taxonomy describes contemporary research, and the language-models category would not appear
in a corpus drawn from the 1990s.

### Text cleaning is deliberately minimal

Whitespace is collapsed. **LaTeX is left intact.** Stripping `$\alpha$` or `\nabla` would remove
real topical signal from mathematics and physics abstracts — in §4.4 the mathematics category is
identified partly by the tokens `mathbb` and `mathcal`. The sentence-transformer tokenisers handle
the markup without help. No stemming, no lowercasing, no stop-word removal: the encoder is
pretrained on natural text and expects it unmodified.

### Stratified sampling, not random sampling

A uniform random draw would leave the rarest archives with too few papers to form a cluster.
The sample is therefore **stratified on the primary archive and proportional**: every archive
is guaranteed to appear, while the real class proportions are preserved. Each archive's rows
are drawn by reservoir sampling (Algorithm R) against a seeded RNG, in the same single pass.

**Proportional, deliberately not balanced.** Flattening the archives to equal size would be
easy and would make the downstream classification task look better. We do not, because the
number of categories is what this project is trying to measure — and a rebalanced sample
would make that number an artefact of our sampling rather than a property of the literature.

The right-hand panel above is the check. Every one of the 20 archives is drawn in proportion
to the eligible population, with a largest deviation of **0.000 percentage points**:

| Archive | Population | Sample | Pop. share | Sample share |
|---|---|---|---|---|
| cs | 761,535 | 53,930 | 35.95% | 35.95% |
| math | 398,604 | 28,228 | 18.82% | 18.82% |
| cond-mat | 176,562 | 12,504 | 8.34% | 8.34% |
| astro-ph | 166,030 | 11,758 | 7.84% | 7.84% |
| physics | 151,290 | 10,714 | 7.14% | 7.14% |
| … | | | | |
| hep-lat | 6,108 | 433 | 0.29% | 0.29% |
| nucl-ex | 5,004 | 354 | 0.24% | 0.24% |

That spread — `cs` is **152×** the size of `nucl-ex` — is the reason macro-F1 is the headline
metric everywhere downstream, and why the classifier weights classes in the loss (§4.5).

| Phase | Papers | Purpose |
|---|---|---|
| **main** | 150,000 | The reported results |
| **pilot** | 10,000 | A subsample of main; matches Arcan's scale for direct comparison |

The pilot is drawn *from* the main sample and inherits its split assignment, so no paper ever moves
between train and test across phases. Validating every step on the pilot first turned a 90-minute
mistake into a 90-second one several times.

### The split is fixed once, before anything else

| Split | main | pilot |
|---|---|---|
| train | 105,000 | 7,022 |
| validation | 22,499 | 1,544 |
| test | 22,501 | 1,434 |

Stratified 70/15/15 on the primary archive, seeded. Every later step reads this split; none of them
recomputes it.

### Reproducibility

The Kaggle dataset is re-published monthly, so "the arXiv dataset" is not a reproducible object. We
pin **version 309** and commit the sampled arXiv IDs to `data/splits/`, which makes the exact
sample recoverable even after the dataset changes underneath us.

```
data/splits/main_ids.csv.gz     150,000 ids + split + clustering flag
data/splits/pilot_ids.csv.gz     10,000 ids + split + clustering flag
```

Everything else follows from `SEED = 42` in `src/config.py`, which is the only place paths and
constants are defined.

A note on where data lives: the raw snapshot (5.65 GB) and the derived embeddings (230 MB) are
written to `%LOCALAPPDATA%`, **not** into the project folder, which sits inside a synced
OneDrive directory. Only small, traceable artifacts — statistics, metrics, sampled IDs, figures —
are committed.

## 4. Model training

This section covers both models the task needs: the unsupervised one that discovers the
categories, and the supervised one that learns to assign them.

### 4.1 Representation

Abstracts are encoded with a frozen sentence-transformer. Nothing is trained here.

| | |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` (384d, max 256 tokens) |
| Input | abstract only, no title |
| Output | L2-normalised float32 |
| Throughput | 150,000 abstracts in 81.7 min on CPU (30.6 rows/s) |

Three choices, each with a reason:

**Frozen, not fine-tuned.** Every study reviewed in §1.2 uses off-the-shelf embeddings for
this task. Fine-tuning an encoder to improve clustering is a separate research problem,
and the one paper that measured it on this exact data found the two *science-specific*
models (SciBERT, SPECTER) lost to the two general-purpose ones.

**Abstract only.** The brief specifies abstracts, and Arcan (2026) found abstract-only
input beat every richer variant they tried.

**L2-normalised.** This makes Euclidean distance and cosine similarity equivalent, which
is what KMeans assumes. Without it, KMeans would be clustering on vector magnitude —
roughly, on how long the abstract is.

### 4.2 Finding candidate groupings

Two algorithms with complementary failure modes are swept, so neither one's assumptions
go unchallenged:

- **KMeans**, k = 3…20, on all 150,000 papers. Assigns every paper; assumes roughly
  convex clusters.
- **HDBSCAN**, `min_cluster_size` from 0.2% to 4% of the corpus. Finds its own number of
  clusters and can refuse to assign a paper.

HDBSCAN is run on a 50-dimensional UMAP reduction, the fix for the failure described in
§1.5. It is also run on a 50,000-paper subsample: on the full sample it reached 16 GB of
paged memory on a 32 GB machine and made no progress in twenty hours. KMeans has no such
problem.

#### The UMAP reduction fixes the noise collapse

Arcan ran HDBSCAN on raw high-dimensional vectors and reported 3–5 clusters with **up to
74% of papers discarded as noise**, concluding that HDBSCAN "performed worst by a wide
margin." With the reduction in place, that collapse does not happen:

| | Arcan (no reduction) | pilot, 10k | main, 50k |
|---|---|---|---|
| Papers left as noise | up to **74%** | 1.0 – 31.4% | **0.0 – 16.8%** |
| Best ARI vs archives | 0.01 – 0.03 | **0.504** | **0.610** |
| Best DBCV | not reported | 0.681 | 0.716 |

Both the noise rate and the agreement with arXiv improve by a large margin, at both
scales, and the best ARI exceeds Arcan's best reported figure of 0.470 — obtained,
unlike theirs, without ever consulting the labels to choose a configuration.

![HDBSCAN sweep](reports/figures/clustering_hdbscan_main.png)

#### But HDBSCAN's cluster count does not survive a change of scale

The pilot alone would have supported a stronger claim. It should not be made. Running the
same `min_cluster_size` *fraction* on the two corpus sizes gives different answers:

| `min_cluster_size` | pilot (10k) | main (50k) |
|---|---|---|
| 0.5% of corpus | **12** clusters | **3** clusters |
| 1% of corpus | **7** clusters | **2** clusters (82% in one) |
| 2% of corpus | 2 clusters | 4 clusters |
| 4% of corpus | 2 clusters | 4 clusters |

The 7-cluster solution that looked so convincing on 10,000 papers does not reappear on
150,000. The count swings with both the parameter and the corpus size, and neighbouring
settings produce degenerate two-cluster partitions in which one cluster holds 82% of the
data.

This is exactly the instability the literature review flagged for this family of methods
(§1.2): reported topic counts that vary wildly on the same data. It means **HDBSCAN
cannot be the instrument that answers "how many categories are there."** What it provides
is a useful independent check on a number chosen some other way, and a demonstration that
the reduction step matters.

So the number is settled by stability instead, and KMeans — whose only free parameter *is*
the number — carries the final taxonomy.

One number is deliberately *not* compared with Arcan: our silhouette scores (0.40–0.59)
dwarf theirs (0.063), but only because ours are measured inside the UMAP reduction, which
compresses clusters by construction. The two figures are not commensurable and no
improvement is claimed there.

### 4.3 Choosing the number of categories

This is the question the brief actually asks, so it gets its own instrument.

#### Why not silhouette

![k sweep](reports/figures/clustering_k_sweep_main.png)

On the 150,000-paper run, silhouette's optimum is **k=4** (0.552) — a solution in which a
single cluster holds **45.6% of the entire corpus**. Silhouette rewards a small number of
well-separated blobs, which is exactly the wrong bias when the question is how finely the
data divides. It is reported, and it decides nothing. (The pilot behaves the same way,
peaking at k=3 with 44.3% in one cluster.)

The two reference levels also disagree, as predicted in H4: ARI against the 20 archives
peaks at k=4, while AMI against the 8 groups peaks at k=3. Coarser reference, coarser
answer. Neither is used for selection — they appear here only to show that no single
external ruler could have settled this either.

#### Stability, and a null that actually works

Stability asks whether a grouping is *reproducible*: cluster two independent 80%
subsamples, measure how much they agree (ARI) on the papers both saw, repeat ten times.

The subtlety is the null. ARI is chance-corrected, so random *labellings* score 0 — but
that is not the relevant comparison, because KMeans will happily carve a featureless blob
the same way twice. The null has to be structureless data of the same shape.

**A first attempt at this failed, and the failure is worth recording.** Shuffling each
embedding coordinate independently preserves every marginal distribution while destroying
the joint structure — a standard trick, and wrong here. UMAP coordinates are strongly
multi-modal, so the product-of-marginals distribution keeps pronounced lumps at the
cross-products of the modes, and KMeans recovers them reproducibly. The "structureless"
null scored **0.9998**. Any margin computed against it measured how fast the null decayed,
not whether real structure existed, and it selected k=16 for no good reason.

The null used instead is **points drawn uniformly on the unit sphere** — the correct shape
for L2-normalised vectors, and genuinely without structure. It scores **0.006–0.014**
across the whole sweep, which is what a null should look like.

#### And a space that actually discriminates

Stability must also be measured somewhere sensible. Inside the UMAP reduction it
saturates: it reaches **1.000 ± 0.000** at k=3, and **17 of the 18** candidate values
clear the threshold, so the rule simply returns whatever the sweep's upper limit happens
to be. `min_dist=0.0` manufactures hard-separated clumps, so what is being measured there
is UMAP's determinism rather than structure in the data. Measured in the embedding space
the same sweep passes only 4 of 18.

The number of categories is therefore decided in the **original embedding space**. UMAP is
used where it is needed — density clustering — and nowhere else.

![Stability across three sweeps](reports/figures/stability_consensus_embedding.png)

#### The selection rule

Ben-Hur et al. require scores that are both high *and* concentrated, so the decision is
made on a **one-standard-deviation lower bound** rather than on the mean. This is not a
cosmetic choice: on the pilot, the mean alone selects k=11 (0.849 ± 0.067) over k=8
(0.942 ± 0.020), which inverts the evidence.

**A note on sample size.** The estimator is expensive — ten repetitions × two subsamples
× two data sets (observed and null) × ten KMeans initialisations is 400 clusterings per
candidate k, 7,200 across a sweep — so each sweep runs on a random subsample capped by
`--max-n` rather than on all 150,000. This is the one result in the report not computed
on the full corpus; clustering, interpretation, agreement and baselines all are. Each
sweep's sample size is in its filename and in its `n_papers` field.

#### Three sweeps, and why one is not enough

The sweep was run three times on independent samples — 10,000, 20,000 and 50,000
abstracts. Each produces a curve that looks clean on its own, and **they disagree about
which k are stable**:

| Sweep | k values whose lower bound clears 0.80 |
|---|---|
| pilot, n=10,000 | 3, 4, 5, **8** |
| main, n=20,000 | 3, 4, 5, 6, 7, **8** |
| main, n=50,000 | 3, 5, 6, 7, **8**, 9, 15, 16, 17, 18 |
| **stable in all three** | **3, 5, 8** |

The disagreement is not noise around a common shape — it is structural. The pilot dips
at k=6 and k=7 and recovers at 8. The 20,000-row sweep collapses cleanly from k=9. The
50,000-row sweep collapses at k=10, then **revives** to a lower bound of 0.93 at k=17.

**Two selection rules were tried and both discarded.** Taking the largest k that clears
the threshold anywhere returns **18** on the 50,000-row sweep, walking past the collapse
into a revival that no other sweep reproduces — the same defect this report criticises in
the UMAP-space measurement, where the rule returns whatever the sweep's upper limit
happens to be. Stopping instead at the first sustained collapse returns **5** on the
pilot, because the pilot's genuine dip at k=6–7 reads as a cliff. Neither rule survives
all three sweeps.

#### The rule that does survive: replication

A number of categories is only meaningful if an independent sample of the same corpus
finds it again. That is the premise the whole method rests on, and it applies to the
sweeps themselves, not only to the subsamples inside one sweep.

> **Rule:** take the largest k whose one-standard-deviation lower bound clears 0.80 in
> **every** independent sweep.

This is strictly more conservative than any single-curve rule: it cannot select a k that
any sweep found unstable, and it is indifferent to where a particular curve is ragged.

> **Answer: 8 categories.** Stable in all three sweeps — lower bound 0.922 at n=10,000,
> 0.946 at n=20,000, 0.893 at n=50,000 — against a structureless null of 0.005–0.008
> throughout. Computed in `stability.py --consensus`, recorded in
> `reports/stability_consensus.json`.

Only k=3, k=5 and k=8 clear in all three. Among reproducible taxonomies the more
informative one is preferred — a three-way split is easy to reproduce and tells you
little — so the answer is the largest of them.

**An independent check agrees.** HDBSCAN — a different algorithm, in a different space,
given no k at all — settles on **7** on the pilot.

**What a single sweep would have told us, wrongly.** Reported from the 20,000-row sweep
alone, this section would have claimed a textbook plateau from k=3 to k=8 with a clean
cliff at k=9. That shape is real in that sample and absent in the other two. The selected
number is unchanged, but the evidence for it is replication across samples, not the shape
of one curve — and a single-sweep report would have made a confident claim it could not
support. This is recorded in §7.4.

### 4.4 What the eight categories are

`interpret.py` describes each category three ways — the words that distinguish it
(c-TF-IDF), the papers nearest its centre, and which arXiv archives land in it. Below is
the taxonomy fitted on all 150,000 abstracts.

| # | Share | Reading | Dominant archive | Distinguishing terms |
|---|---|---|---|---|
| 0 | 19.2% | Algorithms, optimisation and control | `cs` (51%) | algorithm, problem, optimization, control |
| 1 | 7.8% | High-energy theory and gravitation | `hep-ph` (30%) | theory, field, black, mass, gravity |
| 2 | 11.9% | **Language models and NLP** | `cs` (**95%**) | models, language, **llms**, reasoning |
| 3 | 10.7% | Differential equations, numerical analysis | `math` (59%) | equation, solutions, numerical, boundary |
| 4 | 11.4% | Pure mathematics and algebra | `math` (86%) | mathbb, mathcal, prove, algebra, graphs |
| 5 | 8.4% | Astrophysics | `astro-ph` (85%) | stars, galaxies, stellar, emission |
| 6 | 13.6% | Condensed matter and quantum systems | `cond-mat` (50%) | quantum, states, magnetic, spin, optical |
| 7 | 17.0% | Machine learning and computer vision | `cs` (80%) | learning, image, training, neural, deep |

Category names are our reading of the evidence, not model output. Every paper is
assigned, and the largest category holds 19.2% of the corpus — no degenerate blob.

The structure also **replicates**: the pilot, drawn fifteen times smaller and clustered
independently, produced the same eight-way split, including a distinct language-models
category at 10.0% versus 11.9% here.

![UMAP overview](reports/figures/umap_overview_main.png)

![Each category](reports/figures/umap_facets_main.png)

#### The finding: this is not arXiv's taxonomy, and that is the point

Two structural differences, both visible above.

**It splits what arXiv merges.** All of computer science is one arXiv archive. The data
divides it into three categories that share almost no vocabulary: optimisation and
control (0), language models (2), and vision and deep learning (7) — together 48% of the
corpus. Mathematics likewise splits along the applied/pure line into categories 3 and 4.

**It merges what arXiv splits.** arXiv maintains thirteen separate physics archives; the
discovered taxonomy covers all of them with three categories, grouping `hep-ph` with
`hep-th` and `gr-qc` because papers on particle theory, string theory and gravitation
share a vocabulary.

![Composition](reports/figures/cluster_archive_heatmap_main_kmeans_k8_emb.png)

Most striking: **category 2 is one arXiv does not have.** Its top terms are `llms`,
`language`, `reasoning`, and it is 94.6% computer science. There is no arXiv archive for
large language models — the taxonomy predates them, so the work is filed under the
general `cs` umbrella alongside operating systems and computational geometry. The method
recovered a research area the official taxonomy cannot express, and recovered it twice,
independently, at two different scales.

This is the clearest argument for discovering categories rather than inheriting them, and
it is not an argument that any agreement score could have produced.

#### Why the agreement scores are *lower* for the better taxonomy

| Reference level | Homogeneity | Completeness | V-measure |
|---|---|---|---|
| 20 archives | 0.469 | 0.490 | 0.479 |
| 8 groups | 0.526 | 0.357 | 0.425 |

Completeness is the fraction of each arXiv class that stays together. Splitting `cs`
into three necessarily reduces it, because a reference that keeps `cs` whole counts every
split as an error. The 7-cluster HDBSCAN solution scores *higher* on completeness against
archives (0.75 on the pilot) precisely because it leaves the computer-science blob
undivided — and that blob held 43% of the corpus.

A taxonomy is not better for agreeing with arXiv more. This is exactly why §1.4 forbids
using agreement scores to choose between taxonomies, and why the number was settled by
stability instead.

### 4.5 The classifier

With the categories fixed, the supervised task is ordinary: read an abstract, predict one
of the eight.

| | |
|---|---|
| Encoder | `bert-base-uncased` (the brief's HuggingFace filter is `search=bert-base`) |
| Targets | the eight discovered categories |
| Max length | 256 tokens |
| Loss | cross-entropy, inverse-frequency class weights |
| Selection | validation macro-F1, early stopping with patience 2 |

Imbalance is handled with **weights in the loss, not resampling**, so every abstract is
seen exactly once per epoch and the sampling distribution fixed in §3 survives intact.

Fine-tuning runs on a GPU. Measured throughput on this machine puts one CPU run at
about 3.5 hours on the 10,000-paper pilot and roughly 52 hours at full scale; the staged
search in §6 needs ten of them, and the full grid 108. `src/export_training_set.py` writes one self-contained file and
`notebooks/colab_train.ipynb` consumes it on a T4. Everything upstream stays in this
repository under a fixed seed, so the notebook cannot re-derive the labels differently.

## 5. Evaluation

The brief asks for evaluation on a validation set with appropriate metrics. There are two
models here and they are not evaluated the same way, so this section is in three parts:
the taxonomy, the classifier, and why their scores must never be traded against each
other.

### 5.1 Evaluating the taxonomy

Four questions, four instruments. None of them uses a held-out split, because a clustering
has no labels to hold out — the equivalent of held-out evaluation is resampling.

| Question | Instrument | Result |
|---|---|---|
| Is it reproducible? | ARI between independent subsamples, three sweeps | lower bound **0.922 / 0.946 / 0.893** at k=8 |
| Does it replicate? | k stable in *every* sweep | **3, 5, 8** — 8 is the largest |
| Is it better than nothing? | the same, on structureless data | null **0.005–0.008** |
| Is it balanced? | share held by the largest category | **19.2%** |
| Does it cover the data? | fraction of papers assigned | **100%** |
| Is it readable? | c-TF-IDF terms, centroid papers | §4.4 |

The reproducibility figures are the ones that matter, and the point is that there are
three of them. At k=8 two independent 80% subsamples agree well above the threshold in
every sweep — at 10,000, 20,000 and 50,000 abstracts — against 0.005–0.008 for uniform
points on the sphere. The taxonomy is a property of the data, not of any one run; that is
the claim a single sweep could not support, and §7.4 records what happened when we tried.

**What we do not claim.** Internal geometric scores were not used to select anything, for
a reason this project demonstrated twice:

- **Silhouette** peaks at k=4 on the 150k run, where one cluster holds 45.6% of the corpus.
- **DBCV** peaks at 0.716 on an HDBSCAN partition with **two** clusters, one holding 82%.

Both internal measures prefer a small number of fat, well-separated blobs. That is a
reasonable thing for a geometric score to reward and a useless answer to the question the
brief asks. They are reported in `reports/clustering_main_minilm.csv` and they decide
nothing.

### 5.2 Evaluating the classifier

Standard supervised evaluation: train on the training split, select on validation, and
read the test split once at the end.

**Macro-F1 is the headline metric, not accuracy.** Each category is weighted equally
regardless of size, so a model cannot do well by serving the big categories and ignoring
the small ones. The majority-class baseline makes the case concretely: on the earlier
7-category HDBSCAN taxonomy it reached **47.8% accuracy while learning nothing at all**,
because one cluster held 43% of the corpus — at a macro-F1 of 0.092. The reported
8-category taxonomy is balanced enough that the same baseline gets only 19.3% accuracy,
which is itself a point in the taxonomy's favour.

Baselines on the 150,000-paper run, in increasing order of what they know:

| Model | What it uses | Validation macro-F1 | Accuracy |
|---|---|---|---|
| Majority class | nothing | 0.0404 | 0.1929 |
| **TF-IDF + LinearSVC** | word 1–2 grams, char 3–5 grams | **0.9005** | 0.8971 |
| Frozen embeddings + logistic regression | the clustering's own space | *0.9808* | *0.9805* |
| Fine-tuned `bert-base-uncased` | the abstract text | **§6.3** | |

The TF-IDF baseline reaches **0.9005 macro-F1** on validation and 0.897 on test, with
per-category F1 between 0.86 and 0.94 — no weak category. It takes six minutes to fit.

The third baseline is **diagnostic and near-tautological**, and is reported only so that
it cannot be mistaken for a result. A KMeans partition is a Voronoi tessellation of the
embedding space, so a linear model in that same space can recover it almost perfectly by
construction. It measures how cleanly the clustering cut the space, not whether the
categories can be recovered from raw text. Only the TF-IDF and transformer rows speak to
the actual task.

**0.9005 is the number the transformer has to beat, and it is demanding.** It says the
eight discovered categories are already largely separable by vocabulary alone — which is
itself evidence that they are coherent, and sets a high bar for what fine-tuning can add.
The pilot shows the same picture at smaller scale (0.883).

### 5.3 Why the two scores are not interchangeable

This is the methodological rule the whole project is organised around, and it is worth
stating in full because it is easy to violate by accident.

**A classifier's macro-F1 is not evidence that a taxonomy is good.** Suppose classifier A
predicts 8 discovered categories at 0.92 macro-F1 and classifier B predicts arXiv's 8
groups at 0.85. It does not follow that the discovered categories are better. Three
independent reasons:

1. **Circularity.** The discovered labels were produced from the embeddings, so a model
   reading those same embeddings is partly marking its own homework. arXiv's labels were
   assigned by authors and carry information the model has never seen.
2. **Granularity.** Fewer, coarser categories are easier to predict. Taken to its limit,
   a single-category taxonomy scores a perfect macro-F1 and conveys nothing. Any metric
   that rewards this cannot be used to choose a taxonomy.
3. **Asymmetric label noise.** Nearly half of all arXiv papers carry more than one
   subject label (§2), so a "wrong" prediction against arXiv is often not wrong. Cluster
   labels have no such ambiguity by construction. The two tasks have different achievable
   ceilings.

So the rule observed throughout: **classifier metrics are comparable only between runs
that share one label set.** Every macro-F1 in §6 is computed against the same eight
categories, which is what makes the comparisons there meaningful. The taxonomy itself was
judged in §5.1, by instruments that never saw a classifier.

This also rules out a comparison a reader might expect to find here: our macro-F1 against
Arcan's reported 0.923. Their classifier predicts their clustering's labels on their
filtered subset, at a cluster count chosen by maximising agreement with the true labels.
The two numbers are not measuring the same thing, and placing them side by side would be
exactly the error described above.

## 6. Hyperparameter tuning and model selection

### 6.1 A staged search, not a grid

The obvious search is a grid over learning rate × batch size × encoder × seed. That is
4 × 3 × 3 × 3 = **108 fine-tuning runs**, and almost all of it is wasted: the factors are
close to independent, so most of the grid re-measures what the first few cells already
established.

Instead each factor is resolved in turn and the winner carried forward — a **staged**
or coordinate-wise search, **10 distinct runs** instead of 108, for the same decisions:

| Stage | Varies | Values | New runs |
|---|---|---|---|
| 1 | learning rate | 1e-5, 2e-5, 3e-5, 5e-5 | 4 |
| 2 | batch size | 8, 16, 32 | 2 (16 already measured in stage 1) |
| 3 | encoder | bert-base, DistilBERT, SciBERT | 2 (bert-base already measured) |
| 4 | seeds | 42, 1337, 2024 | 2 (42 already measured) |
| | | | **10** |

`tune.py --dry-run` prints 13 commands rather than 10, because the winning configuration
is re-issued at the head of each later stage. Those four invocations share one tag, and
`run_one()` serves the repeats from `reports/classifier_<tag>.json` instead of retraining
— so the printed command count overstates the GPU cost by three runs.

Learning rate goes first because it dominates fine-tuning outcomes. SciBERT is included
despite Arcan finding its *embeddings* were the worst of four for clustering — embedding
quality and fine-tuning quality are different properties, and the comparison costs one
run.

Stage 4 is not tuning. It re-runs the winning configuration under three seeds so the
headline number carries a spread instead of implying a precision it does not have. Arcan
reported no error bars at all; a single fine-tuning run is not a measurement.

**What a staged search gives up.** It assumes the factors are close to separable, and
learning rate and batch size are the one pair here where that is known to be imperfect —
the linear-scaling heuristic says the best learning rate moves with batch size, so
resolving lr at batch=16 and then switching to batch=32 can leave lr slightly stale. A
full grid would catch that; this design does not revisit stage 1 afterwards. Two things
limit the damage: the batch range is narrow (8–32, well under an order of magnitude), and
every run is written to its own JSON, so the interaction is visible in the results table
even though the search did not optimise over it. If the stage-2 winner is not 16, the
honest follow-up is to re-run stage 1 at that batch size — two extra runs, not a new grid.

Random search (Bergstra & Bengio, JMLR 2012) is the other standard answer here and would
be a defensible alternative: with most of the variance coming from one or two factors, it
covers more distinct values per run than a grid. We chose the staged search as the default
because it yields an interpretable per-factor table rather than a scatter of
configurations, which matters more when the budget is ten runs.

#### Both searches are implemented

`tune.py` carries the full grid as well, behind one flag, so the escalation is a
configuration change rather than new code:

```bash
python src/tune.py --phase main --labels kmeans_k8_emb --search staged --dry-run  #  13 commands, 10 distinct runs
python src/tune.py --phase main --labels kmeans_k8_emb --search grid   --dry-run  # 108 runs
```

The grid makes no separability assumption at all — it measures every combination of
learning rate, batch size, encoder and seed, so lr × batch interactions are observed
rather than assumed away. Both paths share the same runner, the same
validation-macro-F1 selection, the same per-run JSON and the same results table.

**Neither has been run.** The grid is 108 fine-tuning runs; at roughly 52 CPU-hours per
run at full corpus scale, it is a GPU-cluster workload, not a laptop one. It is in the
repository so that the comparison between the two designs is concrete, and so that the
step from ten runs to an exhaustive search costs a flag rather than an afternoon.

#### The hyperparameters actually used

The one configuration that was run (§6.3) uses **learning rate 2e-5, batch size 16, 3
epochs, 256 tokens, AdamW with weight decay 0.01**. These are the fine-tuning defaults
recommended in the original BERT paper (Devlin et al., 2019, Appendix A.3: batch ∈ {16,
32}, learning rate ∈ {5e-5, 3e-5, 2e-5}, epochs ∈ {2, 3, 4}), so they are a published
starting point rather than an arbitrary pick.

They also sit inside the search space of the reference paper, which tuned with Optuna
over log-uniform learning rate in [1e-6, 1e-4], batch size in {8, 16, 32} and 2–7 epochs.
Two caveats on that comparison, both worth stating plainly:

- **Arcan reports the search space but never the selected values**, so "the same
  hyperparameters as the paper" is not a claim anyone can make from the paper — including
  us. What can be said is that our configuration lies within the range they searched.
- **We deliberately differ on sequence length.** They truncate at 128 tokens; we use 256,
  because our own EDA shows 256 fits 67.2% of abstracts whole against 20.5% at 128 (§2).
  That is a measured decision, not an inherited one.

**Future work.** The honest next step is to run the staged search, and then the full grid
if the stage-2 winner moves off batch=16 — the one place the staged design's separability
assumption is weakest. Until that compute exists, the reported transformer number is a
single configuration and is labelled as such throughout.

### 6.2 Selection discipline

**Selection uses validation macro-F1 only.** The test split is scored once per run and
reported, never compared during the search. Choosing a configuration by its test score
would turn the test set into a second validation set and the reported number into an
overestimate.

Within a run, the best epoch is likewise chosen on validation macro-F1, with early
stopping after two epochs without improvement.

Every run writes a complete record to `reports/classifier_<tag>.json` — hyperparameters,
per-category precision/recall/F1, the confusion matrix, wall-clock time, and the library
versions — so any row of the results table can be traced to the exact command that
produced it.

### 6.3 Results

**Neither search has been run.** Both need a GPU (§6.4): the staged design is 10 distinct
fine-tuning runs, the full grid is 108. What this section reports is
a single fine-tune, run to completion on CPU so that the transformer figure here is
measured rather than projected — one configuration, one seed, at pilot scale:

| Model | val macro-F1 | test macro-F1 |
|---|---|---|
| majority class | 0.0437 | — |
| tf-idf + LinearSVC | 0.8834 | 0.8780 |
| **`bert-base-uncased`, fine-tuned** | **0.9013** | **0.9008** |
| embeddings + logistic regression *(diagnostic, see §5)* | 0.9726 | — |

7,022 training abstracts, 1,544 validation, 1,434 test, on the pilot's eight discovered
categories. lr 2e-5, batch 16, 3 epochs, 256 tokens, class-weighted loss, seed 42; 349.5
minutes on CPU. Per-epoch validation macro-F1 was 0.8539 → 0.8954 → 0.9013, so the run had
converged rather than been stopped early. The complete record — hyperparameters, per-class
scores, confusion matrix, library versions — is in
`reports/classifier_pilot_bertbase_k8_cpu.json`.

Validation and test agree to within 0.0005. That is expected here rather than impressive:
with a single configuration there was no selection pressure on the validation split to
overfit to.

![Confusion matrix](reports/figures/confusion_matrix_pilot.png)

No category collapses — per-class F1 runs from 0.86 (c1, differential equations) to 0.96
(c3, astrophysics), and the weighted and macro scores are within 0.004 of each other, so
the class weighting did its job on an imbalanced set.

**The errors are concentrated where the taxonomy is genuinely soft.** The three categories
derived from computer science trade mistakes among themselves: 8% of language-model papers
(c2) and 8% of deep-learning papers (c7) are predicted as algorithms/optimisation (c0), and
6% of c2 goes to c7. Astrophysics is the cleanest at 97%, and the physics and mathematics
categories are nearly disjoint from the rest. This mirrors the clustering: the cs/ML region
of the embedding space is one dense mass that k=8 cuts into three, and the classifier
inherits exactly those boundaries.

**What this does and does not establish.** It establishes that the discovered categories
are recoverable from raw abstract text by a transformer at about 0.90 macro-F1 — the
taxonomy is learnable, not an artefact of the embedding space it was built in. It does not
establish H5 as a general claim: +1.8 points over tf-idf, from one seed at pilot scale, is
within the range a seed change could plausibly move. And it replaces none of stages 1–4 —
learning rate, batch size, encoder and seed spread are all still unmeasured.

### 6.4 Why this runs on a GPU

Extrapolating from a measured CPU run on this machine, one fine-tuning run over the
105,000 training abstracts at 256 tokens would take about **52 hours**, and the
ten-run search roughly **three weeks**. The same search on a T4 is a couple of
hours. This is not a CPU workload.

The split is deliberate. Everything that defines the experiment — sampling, splitting,
embedding, clustering, and the choice of eight categories — runs here, seeded and
committed. `src/export_training_set.py` then writes one self-contained file holding the
abstracts, the fixed split and the discovered labels, with a manifest recording its
sha256, class counts and split sizes. The notebook consumes that file and **cannot
re-derive the labels differently**, which is what makes its numbers comparable to
everything else in this report.

## 7. Conclusions

### 7.1 The answer

**Eight categories.**

Selected by stability, across three independent sweeps of the corpus — 10,000, 20,000
and 50,000 abstracts. In each, two independent 80% subsamples are clustered separately
and scored by ARI. **k=8 is stable in all three** (lower bound 0.922, 0.946 and 0.893)
against a structureless null of 0.005–0.008. Only k=3, k=5 and k=8 clear the threshold
in every sweep; eight is the largest, and so the most granular taxonomy the evidence
will carry.

The individual curves disagree about everything else, which is why the rule is
replication rather than curve shape — see §4.3. No arXiv label was consulted in reaching
the number. One independent check agrees: HDBSCAN, a different algorithm in a different
space given no k at all, settles on **7** on the pilot.

The eight categories are listed in §4.4. Every paper is assigned, the largest holds 19.2%
of the corpus, and each is interpretable from its distinguishing vocabulary.

### 7.2 What we found that we did not expect

**The discovered taxonomy is structurally the inverse of arXiv's.** It splits computer
science into three categories and mathematics into two, while covering all thirteen
physics archives with three. arXiv does the reverse: one archive for all of computer
science, one top-level group for all of physics. Neither is wrong — arXiv's taxonomy
reflects how research communities organised themselves historically, and the discovered
one reflects what abstracts actually say.

**One discovered category has no arXiv equivalent.** Category 2 — 11.9% of the corpus,
94.6% computer science, top terms `llms`, `language`, `reasoning` — is large-language-model
research. arXiv has no archive for it, because the taxonomy predates the field, so this
work is filed under general `cs` next to operating systems and computational geometry. The
method recovered it twice, independently, at two scales fifteen times apart.

This is the result that justifies the whole exercise. A classifier trained on arXiv's
labels could never have found it, and no agreement score would have revealed it — by
construction, agreement with arXiv *penalises* discovering a category arXiv lacks.

**Better taxonomies can score worse against the reference.** Splitting computer science
into three lowers completeness against arXiv archives from 0.75 (the undivided
seven-cluster solution) to 0.49, because a reference that keeps `cs` whole counts every
split as an error. Agreement scores measure similarity to arXiv, not quality.

### 7.3 Hypotheses, scored

| | Prediction | Outcome |
|---|---|---|
| **H1** | Silhouette and stability will disagree; prefer stability | **Confirmed.** Silhouette peaks at k=4, where one cluster holds 45.6% of the corpus; DBCV peaks on a two-cluster partition holding 82%. Stability selects 8. |
| **H2** | HDBSCAN after UMAP will not show Arcan's collapse | **Partly confirmed.** The noise collapse is fixed (74% → ≤17%) and ARI improves (0.03 → 0.61). But the cluster count is not stable across corpus size, so HDBSCAN cannot decide the number. |
| **H3** | The categories will not line up with arXiv's 8 groups | **Confirmed**, and more strongly than expected — see §7.2. |
| **H4** | The 8 groups will favour fewer categories than the 20 archives | **Confirmed on the pilot** (AMI peaks at k=4 for groups, ARI at k=6 for archives). On the 150k run both peak early (k=3 and k=4), so the gap narrows at scale. |
| **H5** | Fine-tuned BERT will beat TF-IDF + LinearSVC on macro-F1 | **Weakly supported, not established.** On the pilot, fine-tuned `bert-base` reaches 0.9013 val / 0.9008 test against 0.8834 / 0.8780 for tf-idf + LinearSVC — a win, but of 1.8 points from a single seed. The seed spread that would make it a measurement (stage 4) has not been run. |

### 7.4 What we got wrong along the way

Four errors are worth recording, because each changed a result and none would have been
visible from the final numbers alone.

**A null model that was not null.** Stability was first measured against data produced by
shuffling each embedding coordinate independently. That preserves every marginal while
destroying the joint structure, which sounds right and is wrong here: UMAP coordinates are
multi-modal, so the product distribution keeps clusterable lumps. The "structureless" null
scored **0.9998** and the resulting criterion selected k=16 on noise. Replaced with uniform
points on the unit sphere, which scores 0.007.

**Measuring in a space that cannot discriminate.** Stability measured inside the UMAP
reduction saturates — 17 of 18 candidate values clear the threshold and the rule returns
whatever the sweep's upper limit happens to be. `min_dist=0` manufactures separated clumps,
so what is measured there is UMAP's determinism. The number is decided in the original
embedding space, where 4 of 18 clear.

**Over-reading the pilot, twice.** The pilot's seven-cluster HDBSCAN result was initially
read as evidence that the UMAP fix turns HDBSCAN from the worst algorithm into the best.
The 150k run does not support that: at the same `min_cluster_size` fraction, the pilot's 12
clusters become 3, and its 7 become 2. The noise fix is real and replicates; the cluster
count does not.

The pilot's stability curve misled in the same way. On 10,000 papers, k=6 and k=7 looked
unstable (sd 0.117 and 0.129), which suggested a tidy story — that the algorithm is forced
to invent a boundary at 6 and 7 and finds a real one at 8. On the 20,000-row sweep those
two are among the *most* stable values. The selected number survives; the explanation for
it did not.

**A selection rule that read one curve's shape.** The number was first chosen by taking
the largest k whose lower bound cleared the threshold **in a single sweep**. That sweep
showed a plateau from k=3 to k=8 and a clean cliff at k=9, and the rule returned 8 — and
this report described that shape as "what a well-behaved stability curve looks like".

Re-running on 50,000 abstracts broke it. Stability collapses at k=10 as expected, then
**revives** to a lower bound of 0.93 at k=17, and the rule — having no notion of a first
stable block — returned **18**. That is precisely the defect this report already
criticises in the UMAP-space measurement: a rule that lets the sweep's upper limit decide.
The obvious repair, stopping at the first sustained collapse, was tried and returned **5**
on the pilot, whose genuine dip at k=6–7 reads as a cliff. Both rules were discarded.

The reported rule is replication across independent sweeps: a k is selected only if every
sweep found it stable. That gives **8**, the one answer all three sweeps support, and it
does not depend on any curve's shape.

Two mechanical faults surfaced alongside it. Sweeps differing only in sample size wrote to
the same filename, so one run's curve was briefly left stamped with another's metadata —
sample size is now part of the filename. And the plateau-and-cliff claim had already been
written up from a single sweep before the replication was run, which is the third time in
this project a single sample suggested a tidier story than the data supports. The pattern
is consistent enough to state as a finding in its own right: **on this corpus, any claim
resting on the shape of one curve should be assumed unreplicated until it is checked.**

### 7.5 Limitations

**Category granularity is genuinely ambiguous.** Across all three sweeps, k=3, k=5 and
k=8 are reproducible. Eight is defensible as the most granular of them, but a three- or
five-category taxonomy is also supportable and would be a different, coarser answer to the
same question. We report the criterion rather than claiming a unique truth.

**The stability curve is noisier than one sweep suggests.** Three sweeps at 10,000,
20,000 and 50,000 abstracts agree on k=3, 5 and 8 and disagree about every other value,
including where the collapse begins (k=9 or k=10) and whether stability revives at high k
(it does at n=50,000 only). Ten repetitions per candidate was enough to separate signal
from the null by two orders of magnitude, but not enough to pin the curve's shape. More
repetitions, or sweeps at further sample sizes, would tighten this; the replication rule
is what makes the reported number robust to it in the meantime.

**The reference taxonomy is not ground truth.** 48.3% of arXiv papers carry more than one
subject label and 17.3% span more than one top-level group, so "the" category of a paper
often does not exist. Every agreement score here compares a hard partition against a
reference that is itself soft.

**One encoder, one corpus slice.** Results use `all-MiniLM-L6-v2` on papers from 2016
onward. MPNet is implemented and configured but was not run at scale; on Arcan's evidence
it would shift ARI by roughly 0.03. The 2016 cutoff removes arXiv's compositional drift but
also means the taxonomy describes contemporary research, not the archive's history — the
language-models category would not appear in a corpus drawn from the 1990s.

**Clustering saw the whole sample.** KMeans was fitted on all 150,000 papers, so the
validation and test abstracts contributed to defining the categories the classifier is
scored on. The split was still fixed first and `preprocess.py` reserves a 50,000-paper
train-only clustering subset for the robustness check; the integrity test in
`src/check_integrity.py` verifies it contains no validation or test rows.

**HDBSCAN was swept on a subsample.** On the full 150,000 papers it reached 16 GB of paged
memory without progressing, so its sweep uses 50,000. KMeans used everything.

---

## Repository structure

```
├── src/
│   ├── config.py              # single source of truth: paths, seed, constants
│   ├── download_data.py       # Kaggle snapshot, pinned to version 309
│   ├── eda_scan.py            # streaming corpus statistics
│   ├── eda_year_mix.py        # year x archive / year x group cross-tabs
│   ├── preprocess.py          # stratified sample + fixed 70/15/15 split
│   ├── embed.py               # frozen sentence embeddings, cached
│   ├── cluster.py             # UMAP + KMeans and HDBSCAN sweeps
│   ├── stability.py           # chooses the number of categories
│   ├── interpret.py           # c-TF-IDF terms, examples, arXiv composition
│   ├── baseline.py            # majority / TF-IDF+SVM / embedding+logreg
│   ├── train_classifier.py    # fine-tunes BERT on the discovered categories
│   ├── tune.py                # staged hyperparameter search
│   ├── export_training_set.py # bundles one training set for GPU training
│   └── figures.py             # every figure in this report
├── notebooks/
│   └── colab_train.ipynb      # steps 4-6 on a GPU
├── data/splits/               # committed arXiv ids — pins the exact sample
├── reports/                   # metrics, cluster profiles, figures (committed)
├── requirements.txt
└── README.md                  # this report
```

Large and regenerable artifacts — the 5.65 GB snapshot, the sampled abstracts, the
embeddings — are written to `%LOCALAPPDATA%\arxiv-iris`, outside this folder. Only small,
traceable results are committed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set a Kaggle API token and download the data:

```bash
export KAGGLE_API_TOKEN=...    # Windows: $env:KAGGLE_API_TOKEN="..."
python src/download_data.py
```

The file is 5.65 GB and is cached under `%LOCALAPPDATA%\kagglehub`, deliberately outside the
project folder so it is not synced to cloud storage.

## Reproducing the results

Run in order. Every step caches its output and skips unless `--force` is passed.

```bash
python src/eda_scan.py                 # corpus statistics
python src/eda_year_mix.py             # year x subject mix
python src/preprocess.py               # sample + split (writes data/splits/)

# Validate everything on the 10k pilot first
python src/embed.py      --phase pilot --model minilm
python src/cluster.py    --phase pilot --model minilm
python src/stability.py  --phase pilot --model minilm --space embedding
python src/interpret.py  --phase pilot --model minilm --algorithm kmeans --k 8
python src/baseline.py   --phase pilot --embed-model minilm --labels kmeans_k8_emb

# Then the reported 150k run
python src/embed.py      --phase main  --model minilm
python src/cluster.py    --phase main  --model minilm
python src/stability.py  --phase main  --model minilm --space embedding \
                         --max-n 50000 --skip-hdbscan
python src/interpret.py  --phase main  --model minilm --algorithm kmeans --k 8
python src/baseline.py   --phase main  --embed-model minilm --labels kmeans_k8_emb

python src/check_integrity.py --phase both
python src/figures.py    --phase main  --model minilm --labels kmeans_k8_emb
```

Note the flags. `--labels` and `--embed-model` default to the *pilot's* HDBSCAN
configuration, which is not what this report uses — pass `kmeans_k8_emb` explicitly.
`--space embedding` is the reported setting for stability; `--space umap` reproduces the
saturation described in §4.3. `--max-n` caps the stability sweep's sample size.

Fine-tuning runs on a GPU. Export the training set, then run `notebooks/colab_train.ipynb`:

```bash
python src/export_training_set.py --phase main --embed-model minilm --labels kmeans_k8_emb
python src/tune.py --phase main --embed-model minilm --labels kmeans_k8_emb --dry-run
```

The single CPU run reported in §6.3 — three and a half hours of the fifty-two a full-scale
run would take — is:

```bash
python src/train_classifier.py --phase pilot --embed-model minilm \
    --labels kmeans_k8_emb --encoder bert-base --lr 2e-5 --batch-size 16 \
    --epochs 3 --tag pilot_bertbase_k8_cpu
```
