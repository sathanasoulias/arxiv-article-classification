# Research Article Classification on arXiv

**How many categories are in a corpus of scientific abstracts, and can a model predict them?**

Data: the public [arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(Cornell University, version 309) — 3,224,635 records, 5.65 GB.

---

## Summary

The number of categories is unknown, so it is discovered rather than assumed. We encode
150,000 abstracts with a frozen sentence-transformer, cluster them, and choose the number
of categories by **reproducibility**: a value of k counts only if an independent sample of
the same corpus finds it again. Across three sweeps — 10,000, 20,000 and 50,000 abstracts —
only k = 3, 5 and 8 are stable in all three. **Eight** is the largest, and so the most
granular reproducible taxonomy. No arXiv label is used in reaching that number.

The taxonomy is structurally the inverse of arXiv's: it splits computer science into three
categories and mathematics into two, while covering thirteen physics archives with three.
One of its categories has no arXiv equivalent — **large-language-model research, 11.9% of
the corpus** — because arXiv's taxonomy predates the field. A method anchored to arXiv's
labels could not have found it.

The categories are learnable from raw text. On the 150,000-paper corpus, TF-IDF with a
linear SVM reaches **0.9005 macro-F1** on validation (105,000 train / 22,499 validation).
A fine-tuned `bert-base` reaches **0.9013** on the 10,000-paper pilot — a *different and
smaller* corpus, so the two are not directly comparable. 

| | Section |
|---|---|
| **1** | [The problem](#1-the-problem) |
| **2** | [Approach](#2-approach) |
| **3** | [Data](#3-data) |
| **4** | [Discovering the categories](#4-discovering-the-categories) |
| **5** | [Classification](#5-classification) |
| **6** | [Hyperparameter tuning](#6-hyperparameter-tuning) |
| **7** | [Conclusions](#7-conclusions) |

[PIPELINE.md](PIPELINE.md) traces every phase with its parameters and outputs.
[PRESENTATION.md](PRESENTATION.md) is slide-by-slide speaking notes.

---

## 1. The problem

In ordinary text classification you are given the categories. Here **the number of
categories is unknown and determining it is part of the task**, which makes this two
problems: an open one (how many categories, and what are they?) and a supervised one (can
a model assign an unseen abstract?).

arXiv ships subject labels such as `cs.LG`, and the obvious shortcut is to train on them.
We do not, for three reasons:

- **Authors choose their own labels**, so they are applied inconsistently.
- **52.7%** of sampled papers carry more than one subject label and **17.3%** span more
  than one top-level group — "the" category of a paper often does not exist.
- **The taxonomy is older than parts of the field.** arXiv has no archive for
  large-language-model research. A method anchored to its labels cannot discover a
  category arXiv lacks — and §4.4 shows this corpus contains one.


---

## 2. Approach

![Pipeline](reports/figures/pipeline.png)



### Related work

**Arcan (2026)** is the closest published study — it clusters arXiv papers and trains
classifiers on them, comparing four encoders and three algorithms. We adopt two of its
findings as testable choices: abstract-only input beat richer variants, and general-purpose
encoders beat science-specific ones at clustering.

**It is not the same task.** Arcan selects the number of clusters by maximising
`0.5·ARI + 0.5·NMI` **against arXiv's true labels**, conceding this "may inflate reported
ARI/NMI".

---

## 3. Data

### What the corpus looks like

![Papers per year](reports/figures/eda_papers_per_year.png)

![Abstract length](reports/figures/eda_abstract_length.png)

Three findings shaped the design:

- **The field mix is not stationary.** Computer science grows from a minor archive to the
  largest. Sampling across all years would let the discovered categories encode *when*
  papers were written → the 2016 cutoff.
- **Median abstract is 159 words.** 256 tokens fits 67.2% of abstracts whole; 128 fits only
  20.5% → `MAX_TOKENS = 256`.
- **558,087 papers (17.3%) span more than one top-level group** → the reference taxonomy is
  soft, and every agreement score here compares a hard partition against it.

![Archive distribution](reports/figures/eda_archive_distribution.png)

### Sampling and splitting

![Sampling](reports/figures/sampling_funnel.png)

| Filter | Records | Why |
|---|---|---|
| arXiv snapshot v309 | 3,224,635 | — |
| Submitted 2016 or later | 2,118,113 | removes 24 legacy archive codes and the physics-dominated era |
| Abstract ≥ 20 words | **2,113,230** | drops 4,883 stubs with no topical signal |
| Stratified sample | **150,000** | 7.1% of the eligible frame |

**Stratified on primary archive, proportional — deliberately not balanced.** Balancing
would make the discovered number of categories an artefact of our sampling. The right-hand
panel is the check: every one of the 20 archives drawn in proportion, largest deviation
**0.000 percentage points**. The spread is severe — `cs` is **152×** `nucl-ex` — which is
why macro-F1 is the headline metric and why the classifier weights classes in the loss.

| Split | main | pilot |
|---|---|---|
| train | 105,000 | 7,022 |
| validation | 22,499 | 1,544 |
| test | 22,501 | 1,434 |

The split is computed **before** the clustering subset is drawn, and that subset is
train-only; `check_integrity.py` verifies 0 non-train rows in it. The 10,000-paper pilot is
drawn from the main sample and inherits its split, so no paper moves between train and test.

**Text cleaning is minimal: whitespace only, LaTeX left intact.** Stripping `\mathbb` would
delete real signal — it is one of the two strongest terms identifying the pure-mathematics
category (§4.4).

**Reproducibility.** Kaggle republishes this dataset monthly, so we pin **version 309** and
commit the sampled arXiv IDs to `data/splits/`. Everything else follows from `SEED = 42` in
`src/config.py`. Raw data and embeddings are written to `%LOCALAPPDATA%`, never into the
repository.

---

## 4. Discovering the categories

### 4.1 Representation

| | |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` (384d, max 256 tokens) |
| Input | abstract only, no title |
| Output | L2-normalised float32 |
| Throughput | 150,000 abstracts in 81.7 min on CPU |

**Frozen — nothing is trained here.** **L2-normalised**, which makes Euclidean distance
equivalent to cosine; without it KMeans would partly cluster on abstract length.

### 4.2 Why the usual ways of choosing k fail here

Picking the clustering *algorithm* is not the hard part. KMeans returns any k you ask it for. The hard part is **which k**, and the three standard answers all fail on this corpus.

#### The two internal quality scores are optimal on degenerate partitions

![k sweep](reports/figures/clustering_k_sweep_main.png)

| Criterion | Its optimum | What that partition actually looks like |
|---|---|---|
| **Silhouette** | k=4, score 0.552 | one cluster holds **45.6%** of all papers |
| **DBCV** | 2 clusters, score 0.716 (`min_cluster_size=500`) | one cluster holds **81.7%** |
| *for comparison* | KMeans k=8, silhouette 0.438 | largest cluster **18.5%** |

Select k by silhouette — the textbook move — and you get four categories, one of which is
nearly half of all research. Let HDBSCAN pick its own number and score it by DBCV and you get
two, one holding 81.7%. Both are optimal by their own criterion; both are useless as a
taxonomy.

This is not a defect in the metrics, it is what they measure. Silhouette rewards compact,
well-separated blobs, and the most compact way to partition a continuous space is to cut off a
few tight corners and pool everything else. Scientific abstracts do not arrive in
well-separated blobs — they arrive as a continuum with dense regions. Both scores are recorded
for all 23 configurations in `clustering_main_minilm.csv`; neither decides anything.

#### HDBSCAN's own cluster count does not survive a change of scale

HDBSCAN is the tempting option because it needs no k at all — it infers the number from
density. But the same `min_cluster_size` **fraction** gives a different answer at a different
corpus size:

| `min_cluster_size` | pilot, 10k | main, 150k |
|---|---|---|
| 0.5% | 12 clusters | 3 clusters |
| 1% | 7 clusters | 2 clusters |

Neighbouring settings produce degenerate partitions with **82.7%** of papers in one cluster. A
count that moves by 4× when the corpus grows is not an estimate of how many categories exist;
it is a property of density at a given sample size.

> **What survives, and in what role.** KMeans stays as the **partitioner** — once k is known
> it is the right tool for this geometry. HDBSCAN stays as an **independent check** on a
> number chosen some other way. Neither is the instrument that can answer *"how many
> categories are there?"* — that is §4.3.

### 4.3 Stability — how we chose the number

**The idea in one sentence:** if a set of categories is really there, a *different* sample of
the same literature should turn up the same ones.

#### The intuition

Give two librarians a random half each of the same library. Tell both: *"sort your books into
**8** piles."*

If 8 is a natural number of subjects for that library, they come back with recognisably the
same 8 piles — maths here, astronomy there. Hand a book from one librarian's pile to the
other, and it lands in the matching pile.

Now tell them *"sort into **13** piles."* There is no natural 13, so each has to cut some
subject down the middle to make up the count. They will cut in **different places**. The two
results stop matching.

That is the whole method. **The right k is the one two independent halves agree on.** And
notice what it never needs: a correct answer to compare against. No labels — exactly what the
brief requires.

#### How we measure the agreement

For each candidate k from 3 to 20:

1. Take **two random 80% samples** of the corpus.
2. Cluster each sample **separately** into k groups.
3. Many papers land in both samples. For every *pair* of those papers, ask: did both runs put
   the pair together, or both put it apart? **ARI** scores how often the two runs agree.
4. **Repeat 10 times**, and keep all 10 scores.

**ARI = 1.0** means the two runs produced the same grouping. **ARI = 0.0** means they agreed
no more than random chance. 400 clusterings per candidate k; 7,200 for one full sweep.

#### Reading the score — two traps we fell into

**Trap 1: a high score is not automatically a good score.** KMeans cuts a shapeless cloud the
same way every time, so data with *no structure at all* still scores high. We have to know
what "no structure" looks like before a number means anything. So we built one: random points
on a sphere, the same shape as our L2-normalised vectors. It scores **0.005–0.008**.

So **0.92 does not mean "92% good"** — it means *over 100× what structureless data gets*.
That is the comparison every number below should be read against.

**Trap 2: an average hides a wobble.** A k that scores 0.99 five times and 0.60 five times
averages to a respectable 0.80, but it is not stable — it is erratic. So we never use the
average on its own. We use **average − one standard deviation**, and require *that* to clear
**0.80**. A k only passes if it scores high **and** scores consistently.

#### Doing the sweep once is not enough

![Stability across three sweeps](reports/figures/stability_consensus_embedding.png)

We ran the entire sweep three times, on independent samples of different sizes. Each run's
curve looks convincing on its own — and they contradict each other:

| Sweep | k that clear the bar |
|---|---|
| n=10,000 | 3, 4, 5, **8** |
| n=20,000 | 3, 4, 5, 6, 7, **8** |
| n=50,000 | 3, 5, 6, 7, **8**, 9, 15, 16, 17, 18 |
| **in all three** | **3, 5, 8** |

They disagree about where the collapse starts, about whether 6 and 7 are stable, and about
whether stability comes back at high k. So any rule that reads the *shape* of one curve gives
an answer that depends on which sample you happened to draw. We tried two and threw both away:

- *"Take the largest k that clears the bar"* → **18**, on the third sweep. It walks past the
  collapse at k=10 into a revival at k=17 that nothing else reproduces.
- *"Stop at the first sustained collapse"* → **5**, on the first sweep, whose genuine dip at
  k=6–7 looks like a cliff.

> **The rule we used instead:** the largest k that clears the bar in **every** sweep.
>
> This is *stricter* than either discarded rule, not looser — it can never select a k that any
> one of the three sweeps found unstable.

> ### Answer: 8 categories
> Three values clear the bar in all three sweeps — **3, 5 and 8** — and 8 is the largest. Its
> score at k=8 is **0.922 / 0.946 / 0.893** across the three, against the 0.005–0.008 null.
> Computed by `stability.py --consensus`, recorded in `reports/stability_consensus.json`.

**Why 8 and not 3?** Because among numbers that *all* replicate, the more informative one wins.
A three-way split of all of science is trivially easy to reproduce and tells you almost
nothing. That preference was fixed before we saw the result.

**An independent check agrees.** HDBSCAN — a different algorithm, in a different space, given
no k at all — settles on 7 on the pilot.

**What we claim, and what we don't.** 8 is the output of a rule stated in advance. It is not a
unique truth: 3 and 5 replicate too, so a coarser taxonomy is a defensible answer to the same
question.

Method: Lange, Roth, Braun & Buhmann, *Neural Computation* 16(6), 2004; subsample-pair
estimator from Ben-Hur, Elisseeff & Guyon, PSB 2002.

### 4.4 The eight categories

| # | Share | Reading | Dominant archive | Distinguishing terms |
|---|---|---|---|---|
| 0 | 19.2% | Algorithms, optimisation and control | `cs` (51%) | algorithm, problem, optimization, control |
| 1 | 7.8% | High-energy theory and gravitation | `hep-ph` (30%) | theory, field, black, mass, gravity |
| 2 | 11.9% | **Language models and NLP** | `cs` (**95%**) | models, language, **llms**, reasoning |
| 3 | 10.7% | Differential equations, numerical analysis | `math` (59%) | equation, solutions, numerical, boundary |
| 4 | 11.4% | Pure mathematics and algebra | `math` (86%) | mathbb, mathcal, prove, algebra |
| 5 | 8.4% | Astrophysics | `astro-ph` (85%) | stars, galaxies, stellar, emission |
| 6 | 13.6% | Condensed matter and quantum systems | `cond-mat` (50%) | quantum, states, magnetic, spin |
| 7 | 17.0% | Machine learning and computer vision | `cs` (80%) | learning, image, training, neural |

Every paper is assigned; the largest category holds 19.2% — no degenerate blob. The structure
**replicates**: the pilot, clustered independently at a fifteenth of the scale, produces the
same eight-way split.

The **Reading** column is ours; everything else in the table is generated. The rest of this
section is how those terms were arrived at, and what constrains the reading.

#### How the distinguishing terms are derived

Clustering returns integers. To name cluster 5 we need to know what its 12,640 abstracts
have in common that the other 137,360 do not. Two obvious approaches fail first:

- **Most frequent words in the cluster** → `the`, `we`, `paper`, `results`, `model`. Every
  cluster returns almost the same list.
- **TF-IDF per abstract** → the wrong unit. TF-IDF asks which words are unusual *in one
  document*, so it surfaces idiosyncratic one-offs, not what characterises a group.

**c-TF-IDF changes the unit.** All abstracts in a cluster are concatenated into a single
document, so 150,000 abstracts become **8 cluster-documents**. Asking the ordinary question —
which terms are specific to a document — now gives an answer about the cluster
(Grootendorst, *BERTopic*, 2022, eq. 1):

```
W(t,c) = tf(t,c) · log(1 + A / f(t))
         ───────   ─────────────────
         how much   how much it is NOT
         cluster c  used by the others
         uses t
```

`tf(t,c)` is term *t*'s share of all words in cluster-document *c*; `f(t)` is *t*'s total
count across all eight; `A` is the mean cluster-document length (**1,532,420** words here). A
term used evenly everywhere has a large `f(t)`, so the logarithm shrinks toward zero. A term
concentrated in one cluster has a small `f(t)`, so it survives.

##### What that actually does — cluster 5, real values from the run

| term | count in c5 | tf(t,c) | f(t) across all 8 | idf | **W** | rank |
|---|---|---|---|---|---|---|
| `galaxies` | 6,381 | 0.00497 | 6,490 | 5.469 | **0.02718** | **3** |
| `stellar` | 5,491 | 0.00428 | 5,645 | 5.608 | **0.02399** | **5** |
| `model` | 6,257 | 0.00487 | 83,444 | 2.963 | 0.01444 | 14 |
| `quantum` | 120 | 0.00009 | 33,065 | 3.857 | 0.00036 | 3,437 |

Compare the first and third rows. **`galaxies` and `model` occur almost equally often inside
cluster 5** — 6,381 against 6,257 — so raw frequency cannot separate them. The difference is
everywhere else: **98.3%** of all corpus uses of `galaxies` fall inside this cluster, against
**7.5%** for `model`. The `idf` factor turns that into a 1.9× gap in the final weight, and
`galaxies` ranks 3rd while `model` drops to 14th. `quantum`, which belongs to cluster 6, is
pushed to rank 3,437.

That one comparison is the whole mechanism.

##### Four filters applied before any of it runs

| Filter | Setting | What it removes |
|---|---|---|
| English stop words | 318 | `the`, `of`, `we` |
| Boilerplate list | 73 added | `paper`, `results`, `propose`, `novel`, `approach` — scientific filler that otherwise tops every cluster |
| `max_df` | 0.5 | a term in over half the abstracts cannot distinguish anything |
| `min_df` | 10 | typos and one-off coinages |
| Token pattern | `[a-z][a-z-]{2,}` | digits and symbols — but **keeps `mathbb` and `mathcal`** |

Vocabulary after filtering: **31,801 terms**; the top 12 by `W` become the cluster's profile.

That last filter is deliberate. **We do not strip LaTeX**, because in mathematics abstracts it
is content, not noise — `mathbb` and `mathcal` are the two strongest terms identifying cluster
4, and a standard cleaning pipeline would have deleted the best evidence available.

#### Two independent cross-checks

Terms alone can mislead — `model` means different things in astrophysics and NLP — so
`interpret.py` produces two further views of every cluster:

**Papers nearest the centroid.** Average the 384d vectors of all members, then list the three
papers with the smallest Euclidean distance to that average: the most typical papers in the
cluster. For cluster 5 the first is *"Discovering new B[e] supergiants and candidate Luminous
Blue Variables in nearby galaxies"*.

**arXiv composition.** The cluster × archive cross-tab — `astro-ph` at 84.8% for cluster 5.
This is the **only** point in interpretation where the reference taxonomy is consulted, and it
is used to read a cluster, never to choose or score one.

#### Where our judgement enters, and what constrains it

Three sources — `mass, stars, galaxies, star, stellar, observations…` · astronomy titles ·
`astro-ph` 84.8% — and we write **"Astrophysics"**. That last step is a human reading, and the
only manual step in the entire pipeline.

What makes it checkable rather than arbitrary: **all three sources are generated by code and
committed** (`reports/cluster_profiles_main_minilm_kmeans_k8_emb.md`, one profile per
cluster), and they have to **agree**. A cluster whose terms, centroid papers and archive mix
pointed in different directions would be reported as uninterpretable rather than given a
confident name. None of the eight did.

**And the names are inert.** They appear in no source file; nothing downstream reads them. The
classifier trains on the integer array `labels_minilm_kmeans_k8_emb.npy`, so a name we got
wrong would be a poor description in this report and could not move a single metric in §5.

#### This is not arXiv's taxonomy, and that is the point

**It splits what arXiv merges.** All of computer science is one arXiv archive; the data
divides it into three categories sharing almost no vocabulary — optimisation (0), language
models (2), vision and deep learning (7) — together 48% of the corpus. Mathematics splits
along the applied/pure line.

**It merges what arXiv splits.** Thirteen physics archives are covered by three categories,
grouping `hep-ph` with `hep-th` and `gr-qc` because those papers share a vocabulary.

![Composition](reports/figures/cluster_archive_heatmap_main_kmeans_k8_emb.png)

**Category 2 is one arXiv does not have.** Top terms `llms`, `language`, `reasoning`, 94.6%
computer science. arXiv's taxonomy predates the field, so this work is filed under general
`cs` beside operating systems and computational geometry. The method recovered it twice,
independently, at two scales.

#### Better taxonomies can score *worse* against the reference

| Reference level | Homogeneity | Completeness | V-measure |
|---|---|---|---|
| 20 archives | 0.469 | 0.490 | 0.479 |
| 8 groups | 0.526 | 0.357 | 0.425 |

Splitting `cs` into three necessarily lowers completeness, because a reference that keeps
`cs` whole counts every split as an error. The undivided 7-cluster HDBSCAN solution scores
*higher* (0.75) precisely because it leaves a blob holding 43% of the corpus intact.
Agreement measures similarity to arXiv, not quality — which is why the number was settled
by stability.

#### Is the taxonomy any good? The evidence in one table

A clustering has no labels to hold out, so the equivalent of held-out evaluation is
**resampling**. Everything here comes from §4.3 and §4.4:

| Question | Instrument | Result |
|---|---|---|
| Is it reproducible? | ARI between independent subsamples, three sweeps | **0.922 / 0.946 / 0.893** at k=8 |
| Does it replicate? | k stable in *every* sweep | **3, 5, 8** |
| Better than nothing? | the same measurement on structureless data | null **0.005–0.008** |
| Is it balanced? | largest category's share | **19.2%** |
| Does it cover the data? | papers assigned | **100%** |
| Is it interpretable? | c-TF-IDF terms, centroid papers, archive mix | three agreeing sources per cluster |
| Does it line up with arXiv? | homogeneity / completeness vs 20 archives | 0.469 / 0.490 — and see above for why that is the wrong question |

**Not one of these is a classifier score.** The taxonomy is judged on reproducibility and
interpretability. §5 asks an entirely separate question with its own instruments, and §5.4
sets out why the two are never traded against each other.

---

## 5. Classification

The second half of the brief. Having discovered eight categories without using any labels,
can a model assign an **unseen abstract** to one of them from its text alone?

This is a different question from §4, answered with different instruments. §5.4 sets out why
the two are never traded against each other.

### 5.1 The task and the protocol

| | |
|---|---|
| Target | the 8 discovered categories, as integers in `labels_minilm_kmeans_k8_emb.npy` |
| Input | abstract text only — no title, no arXiv metadata |
| Splits | fixed in §3 **before** clustering: 105,000 / 22,499 / 22,501 |
| Headline metric | **macro-F1** |
| Imbalance | **class weights in the loss**, not resampling |
| Protocol | fit on train, select on validation, read the test split **once** |

**Why macro-F1 and not accuracy.** The categories are unbalanced, and accuracy rewards a
model for ignoring the small ones. One row settles it: the majority-class baseline reaches
**19.3% accuracy** on the main corpus and **0.0404 macro-F1**. Macro-F1 averages per-class
F1 with equal weight, so a model that neglects seven categories cannot hide.

**Why weights rather than resampling.** Resampling changes the effective corpus — it makes
the model see a world with different category proportions from the real one. Weighting
changes only the gradient, leaving the data as it is.

### 5.2 Baselines

Two baselines, on the full 150,000-paper corpus:

| Model | What it uses | Val macro-F1 | Val accuracy |
|---|---|---|---|
| Majority class | nothing | 0.0404 | 0.1929 |
| **TF-IDF + LinearSVC** | word 1–2 grams, char 3–5 grams | **0.9005** | 0.8971 |

Test macro-F1 for the SVM: **0.8970**.

**0.9005 is the number to beat, and it is demanding.** A linear model over surface n-grams
already recovers the categories almost as well as anything could — which is itself evidence
the eight categories are coherent, because incoherent ones would not be separable by
vocabulary.

### 5.3 The transformer

`bert-base-uncased`, fine-tuned: lr 2e-5, batch 16, 3 epochs, 256 tokens, class-weighted
loss, seed 42 (§6.2 on where those values come from). Run at **pilot scale** — 349.5 minutes
on CPU — because a full-scale run projects to roughly 87 CPU-hours (§6.3).

| Model | Corpus | Val macro-F1 | Test macro-F1 |
|---|---|---|---|
| TF-IDF + LinearSVC | pilot, 7,022 train | 0.8834 | 0.8780 |
| **`bert-base-uncased`, fine-tuned** | pilot, 7,022 train | **0.9013** | **0.9008** |

Both rows use **identical data** — the same 7,022 / 1,544 / 1,434 split and the same label
file — so this is the one valid head-to-head comparison in the report. The 0.9005 in §5.2 is
a separate, larger measurement and is **not** comparable with either row.

Per-epoch validation macro-F1: 0.8539 → 0.8954 → **0.9013**, so the run converged rather
than stopping early.

#### The score is not an imbalance artefact

A high score on an unbalanced problem deserves suspicion, so the check is recorded rather
than assumed. All eight categories appear in all three splits, and the largest holds under
22% of any of them:

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| train | 1,385 | 748 | 712 | 675 | 985 | 578 | 854 | 1,085 |
| test | 314 | 150 | 130 | 143 | 169 | 134 | 189 | 205 |
| **test F1** | 0.881 | 0.863 | 0.892 | **0.962** | 0.892 | 0.921 | 0.920 | 0.877 |

Per-class F1 spans 0.86–0.96 — **no category collapses**. Macro (0.9008) and weighted
(0.8976) sit within 0.004, the signature of even per-class performance; had one class been
carrying the result, the two would diverge sharply. And the metric rules the failure out by
construction anyway: on this pilot split the majority baseline gets **21.2% accuracy and
0.0437 macro-F1**.

![Confusion matrix](reports/figures/confusion_matrix_pilot.png)

**The errors concentrate where the taxonomy is genuinely soft.** The three
computer-science-derived categories trade mistakes among themselves: 8% of language-model
papers and 8% of deep-learning papers are predicted as algorithms/optimisation. Astrophysics
is cleanest at 97%, and mathematics and physics are nearly disjoint from everything else.
This mirrors the clustering exactly — the cs/ML region is one dense mass that k=8 cuts into
three, and the classifier inherits those boundaries. The model is not making random mistakes;
it is uncertain precisely where the boundaries are least sharp.

**What this establishes, and what it does not.** It establishes that the discovered
categories are **recoverable from raw text** at about 0.90 macro-F1 — so the taxonomy is
learnable, not an artefact of the embedding space it was built in. It does **not** establish
that BERT beats TF-IDF in general: +1.8 points from a single seed at pilot scale is within
the range a seed change could move. We score that as *weakly supported, not established*.


---

## 6. Hyperparameter tuning

### 6.1 Two searches, both implemented

```bash
python src/tune.py --labels kmeans_k8_emb --search staged --dry-run  #  10 distinct runs
python src/tune.py --labels kmeans_k8_emb --search grid   --dry-run  # 108 runs
```

**Staged** resolves one factor at a time and carries the winner forward — learning rate
(4 values), then batch size, then encoder, then a three-seed spread — giving the same
decisions in 10 runs instead of 108. **Grid** is the full Cartesian product, making no
separability assumption, so learning-rate × batch-size interactions are measured rather
than assumed away. Both share the runner, the validation-macro-F1 selection and the outputs.

Stage 4 is not tuning: it re-runs the winner under three seeds so the headline carries a
spread. A single fine-tuning run is not a measurement.

**Selection uses validation macro-F1 only.** The test split is scored once per run and
reported, never compared during the search.

### 6.2 Hyperparameters used

lr **2e-5**, batch **16**, **3 epochs**, 256 tokens, AdamW, weight decay 0.01,
class-weighted loss, early stopping patience 2. These are the fine-tuning defaults
recommended in the BERT paper (Devlin et al., 2019, Appendix A.3).


### 6.3 Status: neither search has been run

Both need a GPU. Scaling the measured pilot run — 15.9 s/step × 19,688 steps — puts a single
full-scale fine-tune at about **87 CPU-hours** before evaluation, which makes the staged
search roughly three weeks on this machine and the grid far longer.

What *was* run is **one configuration at pilot scale**, reported in §5.3:
`bert-base-uncased` at **0.9013** validation / **0.9008** test macro-F1. That is enough to
show the harness works end to end and to give the report a measured transformer number
rather than a projection. It is not a substitute for the search, and it carries no seed
spread, so it cannot settle whether the transformer genuinely beats the TF-IDF baseline.

**This is the clearest gap against the brief.** Both searches are implemented, both are one
flag apart, and neither has been executed. With GPU budget the order would be: the staged
search first, then the grid if the batch-size winner moves off 16 — the only place the
staged design's separability assumption is weak.


---

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate      # Unix: source .venv/bin/activate
pip install -r requirements.txt
export KAGGLE_API_TOKEN=...                          # Windows: $env:KAGGLE_API_TOKEN="..."
python src/download_data.py
```

Every step caches its output and skips unless `--force` is passed.

```bash
python src/eda_scan.py
python src/eda_year_mix.py
python src/preprocess.py

python src/embed.py      --phase main --model minilm
python src/cluster.py    --phase main --model minilm
python src/stability.py  --phase main --model minilm --space embedding --max-n 50000 --skip-hdbscan
python src/stability.py  --consensus
python src/interpret.py  --phase main --model minilm --algorithm kmeans --k 8
python src/baseline.py   --phase main --embed-model minilm --labels kmeans_k8_emb

python src/check_integrity.py --phase both
python src/figures.py    --phase main --model minilm --labels kmeans_k8_emb
```

`--labels` and `--embed-model` default to the pilot's HDBSCAN configuration, which is not
what this report uses — pass `kmeans_k8_emb` explicitly.

Fine-tuning runs on a GPU. Export the bundle, then run `notebooks/colab_train.ipynb`:

```bash
python src/export_training_set.py --phase main --embed-model minilm --labels kmeans_k8_emb
```

### Repository

```
src/          config, download, EDA, preprocess, embed, cluster, stability,
              interpret, baseline, train_classifier, tune, figures, check_integrity
reports/      metrics as JSON/CSV, cluster profiles, figures
data/splits/  committed arXiv IDs pinning the exact sample
notebooks/    colab_train.ipynb — the same search on a T4
```
