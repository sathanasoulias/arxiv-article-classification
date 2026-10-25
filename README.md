# Research Article Classification — arXiv

Sorting research papers into categories using only their abstracts, when nobody tells us how many
categories there should be. Data: the public
[arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) from Cornell University
(one file, 5.65 GB, about 2.7 million papers).

---

# 1. Literature Review and Methodological Rationale

## 1.1 What the problem actually is

The task sounds like ordinary text classification, but it is not. In ordinary classification you
are given a list of categories and you train a model to pick one. Here the number of categories is
unknown, and finding the right number is part of the job.

That makes this two problems in one:

- **An open question:** how many categories are there, and what are they?
- **A normal machine-learning question:** can a model read a new abstract and assign it to a
  category?

The arXiv data does come with its own subject labels, like `cs.LG` or `math.PR`. These are useful,
but we should not treat them as the single correct answer, for two reasons. First, authors choose
their own labels, so they are inconsistent. Second, most papers carry several labels at once, so
there is no single "true" category per paper.

We therefore split the work into four parts and study each one separately:

1. **Representation** — how do we turn an abstract into numbers a computer can compare?
2. **Discovery** — how do we find natural groups of papers without deciding the groups in advance?
3. **Choosing the number of groups** — how do we pick a number and defend it?
4. **Classification** — how do we train a Hugging Face model to label new, unseen abstracts?

Splitting it this way means each decision gets tested on its own, instead of guessing a whole
pipeline up front and hoping it works.

## 1.2 What the literature says

We read the relevant work before choosing a method. Here is what we found and what we took from
each paper.

### The closest existing study

**Arcan (2026), [arXiv:2601.08841](https://arxiv.org/abs/2601.08841)** — *Triples and
Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents.*

This paper does almost exactly our task: it clusters arXiv papers **and** trains classifiers to
predict arXiv subjects. It compares four ways of representing a paper, four embedding models
(MiniLM, MPNet, SciBERT, SPECTER) and three clustering algorithms (KMeans, GMM, HDBSCAN).

Two findings matter a lot for us:

- Plain abstract text worked **better** than fancier representations that added structured
  knowledge. Simpler was stronger.
- General-purpose sentence models (MiniLM, MPNet) beat the science-specific models (SciBERT,
  SPECTER) at clustering, even though the science-specific ones were trained on research papers.

It reports about **0.923 accuracy and 0.923 macro-F1** for abstract-only classification, averaged
over five runs.

**What we take from it:** do not assume the specialised, expensive option is better. Test it. We
also treat the 0.923 figure carefully — it was measured on a *filtered* subset of arXiv, and we do
not yet know how many categories that involved. A score on 6 categories cannot be compared to a
score on 8 or 150, so we use it as a reference point, not as a target to hit.

### Turning text into numbers

**Reimers & Gurevych (2019), [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)** —
*Sentence-BERT.*

Plain BERT is not built for comparing whole texts. If you feed it two abstracts and compare the
output, the results are poor, because BERT was never trained to make similar texts land near each
other. Sentence-BERT fixes this by training specifically for similarity.

**Singh et al. (2023), [arXiv:2211.13308](https://arxiv.org/abs/2211.13308)** — *SciRepEval*, the
benchmark behind SPECTER2.

Tested scientific document embeddings across 24 tasks and found that the best representation
depends on the kind of task. One embedding is not best at everything.

**What we take from these:** use a model built for similarity, not raw BERT. And since no single
embedding wins everywhere, compare a few rather than picking one on reputation.

### Finding groups without fixing the number

**Grootendorst (2022), [arXiv:2203.05794](https://arxiv.org/abs/2203.05794)** — *BERTopic.*

A four-step recipe: turn documents into vectors, squash those vectors down to fewer dimensions
(UMAP), group them (HDBSCAN), then describe each group by the words that make it distinctive
(class-based TF-IDF). The appeal is that HDBSCAN finds the number of groups by itself and can mark
odd documents as outliers instead of forcing them somewhere.

**What we take from it:** this is our main discovery method, and the word-based descriptions are how
we make clusters readable by a human.

**One important warning.** Published work using this approach shows it tends to break data into far
too many groups. Reported topic counts average around 67 where older methods find 15–22 on the same
data. In one reported case it produced **3,561 topics on data that humans had labelled with 4**, and
in several datasets **more than half of all documents were dumped into the outlier bin**.

So we will not simply run HDBSCAN and announce whatever number it returns. That number depends
heavily on its settings. We treat it as one piece of evidence and always report what fraction of
papers were left as outliers.

### Deciding how many groups is right

**Lange, Roth, Braun & Buhmann (2004)**, *Stability-Based Validation of Clustering Solutions*,
Neural Computation 16(6):1299–1323.

The idea is simple and strong. If a grouping reflects something real in the data, you should get
roughly the same grouping from a different sample of that data. If it is an accident of the
algorithm, it will come out differently each time. So: take a random part of the data, group it,
repeat, and measure how much the groupings agree.

**What we take from it:** this is our main tool for choosing the number of categories. Standard
geometric scores like silhouette tend to favour whatever number happens to look tidy; stability asks
the better question, which is whether the grouping is *reproducible*.

### Earlier work on this dataset

**Scharpf et al. (2020), [arXiv:2005.11021](https://arxiv.org/abs/2005.11021)** — JCDL.

An early study of classifying and clustering arXiv content, including how to handle mathematical
notation. It used pre-transformer methods (doc2vec, TF-IDF) on 4,900 documents and reached 82.8%
classification accuracy and 69.4% cluster purity.

**What we take from it:** mainly historical context and a baseline to beat. We are not implementing
its methods, so it informs the problem rather than the solution.

### Our top three

The brief asks for three papers referencing the implementation we will use. Our picks, and why:

| # | Paper | Why |
|---|---|---|
| 1 | **Arcan (2026)** | Closest published version of this exact task; sets our representation and classification choices |
| 2 | **Grootendorst (2022)** — BERTopic | The category-discovery method we implement |
| 3 | **Lange et al. (2004)** | Answers the assignment's central question: how to choose the number of categories |

Supporting references: Sentence-BERT and SciRepEval/SPECTER2 for the representation comparison,
SciBERT and BERT for the classifier, Scharpf et al. for problem history.

### The main conclusion from reading all this

We should not start by committing to one embedding model, one clustering algorithm, or one number of
categories. The literature disagrees with itself on all three, and the paper closest to our task
found that the *simpler* option won. So we write down testable predictions and run a controlled
comparison instead of trusting reputation.

## 1.3 The plan

Six stages. Each produces something concrete.

**Stage 1 — Understand the data.** Abstract lengths, how often each subject appears, how many
labels papers carry, how imbalanced the classes are, and how this changes over time. Fix the
train/validation/test split once, here, so every later experiment uses the same split.

**Stage 2 — Compare representations.** Three options: TF-IDF (counting words, our cheap baseline),
MiniLM (general-purpose sentence model), and SPECTER2 (science-specific). Keep the clustering method
and the data fixed so the only thing changing is the representation.

**Stage 3 — Find groups.** Run KMeans across a range of candidate numbers, since KMeans is simple
and reproducible. Then compare against BERTopic with HDBSCAN, which picks its own number.

**Stage 4 — Choose the number of categories.** Judge each option on four things:

| What we measure | How | The question it answers |
|---|---|---|
| Separation | Silhouette score | Are the groups actually distinct? |
| Stability | Agreement across repeated samples | Would we get this again from different data? |
| Interpretability | Top words and example abstracts per group | Does the group make sense to a human? |
| Coverage | Group sizes and outlier percentage | Does it classify most papers, or give up on them? |

Only after comparing all four do we pick a number and report it.

**Stage 5 — Compare with arXiv's own labels.** For each group we discovered, look at which arXiv
subjects its papers carry. This tells us whether a group matches an existing subject, merges several,
or splits one apart. Because papers have multiple labels, we report label *distributions* per group
rather than pretending each paper has one true class.

**Stage 6 — Train the classifier.** Fine-tune a Hugging Face model on abstracts and measure it on
held-out data.

### Keeping the scope realistic

The full grid of every option above would take weeks of GPU time on 2.7 million papers. We cut it
deliberately:

- Work from one fixed stratified sample of about 150,000 abstracts, not the full dataset
- Compute each set of embeddings **once** and cache them
- Compare the three representations using KMeans only, then carry the winner forward
- Test HDBSCAN against that winner only
- Run stability checks with 5 resamples on a 20,000-paper subset
- Train the classifier on **one** label set, comparing 3 model variants

### What the classifier predicts

The brief can be read two ways, so we state our decision plainly.

**We train on arXiv's existing subject labels**, collapsed to top-level groups. These are real,
externally-assigned labels, so accuracy figures mean something and can be compared to published
work.

Training on our discovered cluster labels instead is a *secondary* experiment. If we report it, we
describe the result as **agreement with our own clustering**, not as classification accuracy. The
reason: cluster labels are produced from the embeddings themselves, so a model re-learning them is
partly marking its own homework. Scores from the two setups are not comparable, because the number
of classes, the difficulty of the boundaries, and the amount of label noise all differ.

## 1.4 What we expect to find

Written as predictions we can be wrong about.

- **H1:** Sentence embeddings (MiniLM) will produce better-separated clusters than TF-IDF, measured
  by silhouette and by AMI against arXiv primary labels at the same number of clusters.
- **H2:** Science-specific embeddings (SPECTER2) will **not** beat general-purpose MiniLM on cluster
  separation. This follows Arcan (2026); we are testing whether it holds at our scale.
- **H3:** Stability and silhouette will disagree about the best number of categories. Where they do,
  we prefer stability, and we say so.
- **H4:** HDBSCAN will return many more groups than the stability analysis supports, and will leave a
  large share of papers as outliers.
- **H5:** Fine-tuned BERT will beat a TF-IDF + LinearSVC baseline on macro-F1. If it does not,
  something is wrong with our training setup rather than with the baseline.

### How we score things, and why

We report **macro-F1**, not accuracy. The classes are very unbalanced — physics papers hugely
outnumber economics papers — and accuracy hides this. Published results on arXiv's 8 top-level
classes show the problem clearly:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Naive Bayes (TF-IDF) | 0.89 | 0.65 |
| LinearSVC (TF-IDF) | 0.82 | 0.82 |

Naive Bayes looks better on accuracy and is much worse in reality: it does well on the big classes
and poorly on the small ones. LinearSVC is our baseline to beat, and it is almost free to run.

For comparing groupings we use **AMI and ARI** rather than raw NMI. Raw NMI drifts upward as the
number of clusters grows, which would unfairly reward splitting the data into more groups. AMI and
ARI correct for chance.

## 1.5 What this study contributes

A reproducible, literature-informed comparison of ways to represent and group scientific abstracts.
We report a number of categories together with the criteria that justify it, show how those
categories relate to arXiv's existing taxonomy, and measure how well a transformer classifier
generalises to unseen abstracts.

We also document the limits. Category granularity is genuinely ambiguous, scientific fields overlap,
and discovered clusters are not the same kind of object as human-assigned subject labels. Where our
evidence does not settle a question, we say so rather than picking the flattering number.

> Sections 1.2–1.4 describe a planned methodology. Once experiments are run, we will separate
> decisions supported by prior work from decisions supported by our own results.

---

## 2. Exploratory data analysis

_Pending._

## 3. Preprocessing

_Pending._

## 4. Model training

_Pending._

## 5. Evaluation

_Pending._

## 6. Hyperparameter tuning and model selection

_Pending._

## 7. Conclusions

_Pending._

---

## Repository structure

```
├── data/                 # empty; dataset is cached outside OneDrive (see Setup)
├── notebooks/            # analysis notebooks, run in order
├── src/                  # reusable modules
├── reports/figures/      # generated plots
└── README.md             # this report
```

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
