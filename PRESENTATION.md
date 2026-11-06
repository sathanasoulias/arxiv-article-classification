# Presentation source material

Slide-by-slide notes for presenting this work to Iris.ai. Each slide carries the
headline to land, the evidence to show, speaker notes, and the question a reviewer is
likely to ask with the answer ready.

`PIPELINE.md` is the technical trace; this is what to say about it.

**Three things to land, in priority order:**

1. The number of categories was chosen **without ever looking at arXiv's labels** — by
   reproducibility across three independent samples, with a null model and a stated
   rule. **8.**
2. The method found a category **arXiv does not have**: large language models, 11.9% of
   the corpus.
3. The rejected approaches are reported as carefully as the accepted one — including
   three of our own errors.

Point 3 is what separates this from a notebook that got a number. Do not cut it for time.

---

## Slide 1 — The question

**Headline:** Classify research articles into an unknown number of categories — and
report the best possible number.

**Speaker notes.** The brief says the number is unknown. That single word determines the
whole design: it rules out training on arXiv's existing labels, because then the number
would be 20 (archives) or 8 (groups) by assumption rather than by evidence. Iris
confirmed this reading — discovering meaningful categories is part of the objective,
with the methodology left to us.

So the work splits in two, and the split is the backbone of this talk:

| Track | Question | Instrument |
|---|---|---|
| **Discovery** | How many categories are there, and what are they? | clustering + stability |
| **Classification** | Can a model predict them from raw text? | fine-tuned transformer |

**The discipline that holds the whole thing together:** these two are measured by
different instruments and **never traded against each other**. A taxonomy is not better
because a classifier scores higher on it — a trivially easy taxonomy would score best of
all.

**If asked — "why not just use arXiv's categories?"**
Because the brief says the number is unknown, and because arXiv's taxonomy has known
defects for this purpose: over half of papers carry more than one label, 17.3% span more
than one top-level group, and it has no archive for large-language-model research. We
use it to *interpret* and *validate*, never to decide.

---

## Slide 2 — Approach in one picture

**Headline:** Ten phases, every one seeded, cached and reproducible from a committed
sample.

**Show:** the pipeline table from `PIPELINE.md`.

```
download → EDA → preprocess → embed → cluster → CHOOSE k → interpret
                                                     ↓
                              baselines → classifier → verify & export → tune
```

**Speaker notes.** Two corpora run through it: a 10,000-paper pilot to validate the
pipeline, and the 150,000-paper main corpus that produces every reported number. The
pilot is not a draft of the result — it is an *independent replication* at a fifteenth of
the scale, and where the two disagree, the main corpus wins and the disagreement gets
reported.

**If asked — "is it reproducible?"**
The Kaggle dataset version is pinned (309, republished monthly), the sampled arXiv ids
are committed to git, every step is seeded, and `check_integrity.py` verifies the
invariants with a non-zero exit code on failure.

---

## Slide 3 — The data

**Headline:** 3.2M papers in; 150,000 sampled, stratified, split before anything else
touched them.

**Numbers for the slide:**

| | |
|---|---|
| Records scanned | 3,224,635 |
| Eligible after filters (2016+, ≥20 words) | 2,113,230 |
| Sampled | **150,000** |
| train / val / test | 105,000 / 22,499 / 22,501 |
| Archives / groups represented | 20 / 8 |
| Imbalance | cs 53,930 → nucl-ex 354 (**152×**) |
| Multi-label papers | **52.7%** |
| Median abstract | 159 words |

**Speaker notes — three decisions worth defending out loud:**

**The 2016 cutoff.** arXiv's field mix is not stationary — computer science grows from a
minor archive to the largest one. Sampling all years would mean the discovered
categories partly encode *when* papers were written rather than what they are about.
2016 also removes all 24 legacy archive codes.

**Proportions preserved, not rebalanced.** Stratified on primary archive so every
archive appears — but deliberately *not* flattened to equal sizes. Rebalancing would
make the discovered number of categories an artefact of our sampling.

**The split comes first.** Computed before the clustering subset is drawn, and that
subset is train-only. Verified at zero validation/test rows.

**If asked — "152× imbalance, isn't that a problem?"**
It is the reason macro-F1 is the headline metric everywhere. One number makes the case:
the majority baseline scores **19.3% accuracy but 0.0404 macro-F1**.

---

## Slide 4 — Encoding

**Headline:** Frozen sentence embeddings. Nothing is trained here.

**Numbers:** `all-MiniLM-L6-v2`, 150,000 × 384, 256 tokens, L2-normalised.

**Speaker notes.** Three choices to name:

- **Frozen, not fine-tuned.** Every paper in the literature review uses off-the-shelf
  embeddings for this; fine-tuning an encoder *for clustering* is a separate research
  problem.
- **Abstracts only, no titles.** The brief specifies abstracts, and the prior work found
  abstract-only input beat every richer variant.
- **L2-normalised**, which makes Euclidean distance equivalent to cosine — what KMeans
  assumes. Unnormalised vectors would let KMeans cluster partly on magnitude.

**256 tokens** fits 67.2% of abstracts whole; 128 would fit only 20.5%.

**If asked — "why MiniLM and not a science-specific model?"**
Prior benchmarking on this exact task put both science-specific models *last*:
all-mpnet 0.470, MiniLM 0.442, SPECTER 0.415, SciBERT 0.304. MiniLM is ~5× faster than
MPNet at close to the same quality. MPNet is implemented but was not run at scale — we
list that as a limitation rather than claiming it would not matter.

---

## Slide 5 — Why the standard answer fails

**Headline:** Both internal cluster-quality scores reward degeneracy on this data.

**Show this table — it is the most persuasive slide in the deck:**

| Configuration | "Best" score | Largest cluster |
|---|---|---|
| KMeans k=4 | silhouette **0.5516 — the optimum** | **45.6%** of the corpus |
| HDBSCAN mcs=100 | DBCV **0.631** | **82.7%**, only 2 clusters |
| KMeans k=8 | silhouette 0.4383 | 18.5% |

**Speaker notes.** If you select k by silhouette — the textbook move — you get four
categories, one of which holds nearly half of all research. If you let HDBSCAN find its
own number by DBCV, you get two, one holding 82.7%. Both are *optimal* by their own
criterion and both are useless as a taxonomy.

This is not a failure of the metrics; it is what they measure. Silhouette rewards
compact, well-separated blobs, and the most compact way to partition a continuous
embedding space is to cut off a few tight corners and dump the rest together.

**This slide earns the next one.** Without it, stability selection looks like an exotic
choice. With it, it is the obvious response to a demonstrated problem.

**If asked — "did you also try the elbow method / gap statistic?"**
The sweep records silhouette, DBCV, noise fraction, largest-cluster share and cluster
sizes for all 23 configurations — the degeneracy is visible in every one of them. We
chose stability because it answers a different question entirely, not a better-tuned
version of the same one.

---

## Slide 6 — How we chose the number

**Headline:** If the categories are real, two different samples should find the same
ones.

**Speaker notes.** The method needs no labels at all, which is exactly what the brief
requires. For each candidate k: draw two independent 80% subsamples, cluster each
separately, measure agreement (ARI) **on the overlap only**, repeat 10 times, keep mean
and standard deviation. If k is right, both runs recover the same structure. If k is
wrong, the algorithm must invent boundaries — and invented boundaries land somewhere
different each time.

Lange, Roth, Braun & Buhmann, *Neural Computation* 2004; subsample-pair estimator from
Ben-Hur, Elisseeff & Guyon, PSB 2002.

**Two refinements worth a sentence each:**

**The null.** ARI is chance-corrected, so random *labellings* score 0 — but that is the
wrong null. KMeans will carve a featureless blob the same way twice, so the real question
is whether the data is more reproducible than *structureless data of the same shape*. Our
null is points drawn uniformly on the unit sphere — the correct shape, because the real
vectors are L2-normalised. It scores **0.007**.

**The rule.** Scores must be high *and* concentrated, so selection uses a
**one-standard-deviation lower bound**, not the mean. And — the part that took three
attempts to get right — a k counts only if **every** independent sweep found it stable.
Among those, take the largest.

**If asked — "isn't this expensive?"**
400 clusterings per candidate k, 7,200 across one sweep, and we ran three. That cost is
why each sweep runs on a capped subsample — the one result in the report not computed on
all 150,000. Sample size is in every filename and every output record.

---

## Slide 7 — The answer

**Headline:** **8 categories** — the largest number that *every* independent sweep found
stable.

**Show:** `reports/figures/stability_consensus_embedding.png` — all three sweeps on one
axis, with the shared-stable values shaded.

| Sweep | k values that clear the bar |
|---|---|
| n=10,000 | 3, 4, 5, **8** |
| n=20,000 | 3, 4, 5, 6, 7, **8** |
| n=50,000 | 3, 5, 6, 7, **8**, 9, 15, 16, 17, 18 |
| **all three** | **3, 5, 8** |

**Speaker notes.** We ran the sweep three times on independent samples. Each curve looks
decisive on its own — and they disagree about almost everything: where the collapse
starts, whether k=6 and k=7 are stable, whether stability revives at high k.

Three values survive in all three: 3, 5 and 8. **Eight is the largest**, and among
reproducible taxonomies the more informative one is preferred — a three-way split is
trivially easy to reproduce and tells you almost nothing.

k=8's lower bound: **0.922 / 0.946 / 0.893** across the three sweeps, against a
structureless null of 0.005–0.008. Two orders of magnitude above noise, every time.

**Say this plainly, it is the strongest line available:** the number does not rest on the
shape of any one curve. It rests on three independent samples agreeing.

**Be honest and do not let them find it:** eight is not forced. k=3 and k=5 are also
reproducible, so a coarser taxonomy is a defensible answer to the same question. What we
claim is the criterion, stated in advance, not a unique truth.

**If asked — "why not k=3, it scores highest?"**
Because the rule was fixed before the result: among reproducible options, prefer the more
informative. A three-way split of all of science is stable and useless.

**If asked — "why three sweeps rather than one bigger one?"**
Because a single sweep cannot tell you whether its own shape replicates — and on this data
it does not. The first sweep showed a textbook plateau and cliff. Running it again at a
larger sample showed that plateau was partly specific to that sample. See Slide 13.

**If asked — "isn't picking the intersection post-hoc?"**
The threshold and the lower-bound test were fixed in advance. What changed is that we
stopped trusting one curve's shape and required agreement across samples — which is
strictly *more* conservative. It cannot select a k that any sweep found unstable.

---

## Slide 8 — What the eight categories are

**Headline:** Every paper assigned, no cluster degenerate, each interpretable from its
own vocabulary.

| | Category | Share | Distinguishing terms | Dominant archive |
|---|---|---|---|---|
| c0 | Algorithms & optimisation | 19.2% | algorithm, problem, performance | cs 51% |
| c1 | High-energy theory | 7.8% | theory, field, black, mass | hep-ph 30% |
| c2 | **Language models** | 11.9% | models, language, **llms**, reasoning | cs **95%** |
| c3 | Differential equations | 10.7% | equation, solutions, numerical | math 59% |
| c4 | Pure mathematics | 11.4% | **mathbb**, **mathcal**, prove, algebra | math 86% |
| c5 | Astrophysics | 8.4% | stars, galaxies, stellar | astro-ph 85% |
| c6 | Condensed matter | 13.6% | quantum, states, magnetic, spin | cond-mat 50% |
| c7 | Machine learning & vision | 17.0% | learning, image, training, neural | cs 80% |

**Speaker notes.** Largest 19.2%, smallest 7.8% — nothing degenerate. Names come from
three independent sources: c-TF-IDF distinguishing terms (BERTopic, eq. 1), the abstracts
nearest each centroid read by hand, and the arXiv composition of each cluster.

**A small detail worth mentioning** — it shows the preprocessing was thought about: we
deliberately did *not* strip LaTeX, because in mathematics abstracts it is content, not
noise. `mathbb` and `mathcal` are the two strongest terms identifying pure mathematics.
A standard cleaning pipeline would have deleted them.

**If asked — "how do you know these are meaningful and not just clusters?"**
Three independent lines of evidence per cluster, and they agree. Also: both the pilot and
the main corpus produce recognisably the same taxonomy at fifteen times' difference in
scale.

---

## Slide 9 — The finding

**Headline:** The method recovered a category arXiv does not have.

**The number:** **c2 — large language models. 11.9% of the corpus. 94.6% computer
science. Terms: `llms`, `language`, `reasoning`.**

**Speaker notes.** arXiv has no archive for LLM research, because the taxonomy predates
the field. This work is filed under general `cs`, next to operating systems and
computational geometry. The method found it twice, independently, at two scales fifteen
times apart.

**This is the result that justifies the whole exercise.** A classifier trained on arXiv's
labels could never have found it — by construction it can only reproduce categories that
already exist. And no agreement score would have revealed it, because agreement with
arXiv actively *penalises* discovering a category arXiv lacks.

**The structural finding, worth one more line.** The discovered taxonomy is roughly the
*inverse* of arXiv's: it splits computer science into three categories and mathematics
into two, while covering all thirteen physics archives with three. arXiv does the
reverse — one archive for all of computer science, one group for all of physics. Neither
is wrong. arXiv's reflects how communities organised themselves historically; ours
reflects what abstracts actually say now.

---

## Slide 10 — Agreement with arXiv, and why it is not the score

**Headline:** We report agreement. We never optimised it.

| Reference | Homogeneity | Completeness | V-measure |
|---|---|---|---|
| 20 archives | 0.469 | 0.490 | 0.479 |
| 8 groups | 0.526 | 0.357 | 0.425 |

**Speaker notes.** Homogeneity and completeness are reported separately rather than
folded into V-measure, because they fail in *opposite* directions and the distinction is
the finding: clusters can be pure but fragmented, or coherent but conflated.

**The point to make, and it is counter-intuitive:** a better taxonomy can score *worse*
here. Splitting computer science into three categories drops completeness against arXiv
archives from 0.75 to 0.49 — because a reference that keeps `cs` whole counts every split
as an error. Agreement measures *similarity to arXiv*, not quality.

This is also the clean contrast with the prior work. Arcan (2026) selected the number of
clusters by maximising `0.5·ARI + 0.5·NMI` against arXiv's true labels — and conceded it
"may inflate reported ARI/NMI". That is circular when the number is unknown: it uses the
answer to find the answer.

**If asked — "0.47 agreement sounds low, is that good?"**
It is the wrong question, and that is the point of the slide. The reference is itself
soft — over half of papers carry multiple labels — and we are not trying to reproduce it.
If we scored 0.95 against arXiv we would simply have rediscovered arXiv.

---

## Slide 11 — Can the categories be learned from text?

**Headline:** Yes — and a cheap baseline already gets most of the way.

**150,000 papers, 8 discovered categories:**

| Model | val macro-F1 | val accuracy |
|---|---|---|
| majority class | 0.0404 | 0.1929 |
| tf-idf + LinearSVC | **0.9005** | 0.8971 |

**Speaker notes.** The first row makes the metric argument in one line: 19% accuracy,
0.04 macro-F1. That gap is why macro-F1 is the headline everywhere.

**Do not put the embeddings-plus-logistic-regression number on a slide.** `baseline.py`
computes it and it scores ~0.98, which looks like the best result in the project and is
not a result at all: the clusters were *defined* in that embedding space, and a KMeans
partition is a Voronoi tessellation of it, so the task is linearly separable there by
construction. It measures how cleanly the clustering cut the space, not whether the
categories generalise from raw text.

**If someone finds it** — in the repo, the JSON, or `baseline.py` — that is the answer,
and giving it quickly is worth more than the number ever was. The report says the same in
§5.2.

---

## Slide 12 — The transformer

**Headline:** Fine-tuned `bert-base` reaches 0.9013 validation / 0.9008 test macro-F1.

**Configuration:** lr 2e-5, batch 16, 3 epochs, 256 tokens, class-weighted loss, seed 42,
best epoch chosen on validation macro-F1, test scored once.

Per-epoch validation macro-F1: 0.8539 → 0.8954 → **0.9013**.

**Show:** `reports/figures/confusion_matrix_pilot.png`

**Speaker notes.** Per-class F1 runs 0.86 to 0.96 — no category collapses — and macro and
weighted scores sit within 0.004, so the class weighting worked. Weights go in the loss,
not into resampling: resampling changes the effective corpus, weighting changes only the
gradient.

**The confusion matrix tells a story worth telling.** Errors concentrate among the three
computer-science-derived categories: 8% of language-model papers and 8% of deep-learning
papers are predicted as algorithms/optimisation, and 6% of language models goes to
machine learning. Astrophysics is cleanest at 97%; physics and mathematics are nearly
disjoint from everything else.

That mirrors the clustering exactly. The cs/ML region of the embedding space is one dense
mass that k=8 cuts into three, and the classifier inherits those boundaries. The model is
not making random mistakes — it is uncertain precisely where the taxonomy is genuinely
soft.

**State the scope before they ask.** This ran on the **pilot** (7,022 training abstracts),
one configuration, one seed, on CPU — 5h49m. A full-scale run projects to ~87 hours on
CPU and the ten-run search to roughly three weeks, which is why it needs a GPU.

**If asked — "so does BERT beat the baseline?"**
0.9013 against 0.8834 on the pilot — a win of 1.8 points, from a single seed. We score
that hypothesis as **weakly supported, not established**. The seed spread that would make
it a measurement has not been run. A single fine-tuning run is not a measurement.

---

## Slide 13 — What we got wrong

**Headline:** Four errors, each of which changed a result, none visible from the final
numbers.

**Speaker notes — this slide is a differentiator, not a confession.** Any reviewer has
seen a dozen submissions that report only what worked. Three bugs found by our own checks:

**1. A null model that was not null.** The first "structureless" baseline shuffled each
embedding coordinate independently. That preserves every marginal while destroying the
joint structure — which sounds right and is wrong here, because the coordinates are
multi-modal, so the product distribution stays clusterable. It scored **0.9998**, and the
criterion built on it selected **k=16 on noise**. Replaced with uniform points on the unit
sphere, which scores 0.007.

**2. Measuring in a space that cannot discriminate.** Measured inside the UMAP reduction,
stability saturates — 17 of 18 candidates clear the threshold and the rule returns
whatever the sweep's upper limit happens to be. `min_dist=0` manufactures separated
clumps, so what is measured there is UMAP's own determinism. The number is decided in the
original embedding space, where 4 of 18 clear.

**3. Over-reading the pilot — twice.** (a) The pilot's HDBSCAN result looked like evidence
that reducing with UMAP turns HDBSCAN from worst algorithm to best; at 150,000 papers the
cluster counts do not hold, though the noise fix does. (b) The pilot showed k=6 and k=7 as
*unstable*, making k=8 look like an island of reproducibility between noisy neighbours —
a tidy story. It did not replicate: at scale those are among the *most* stable values.
**The selected number was right, but the reasoning we first gave for it was wrong.**

**4. A selection rule that read one curve's shape — the one that mattered most.** The
number of categories was first chosen by taking the largest k clearing the threshold *in
a single sweep*. That sweep showed a plateau from k=3 to k=8 and a clean cliff at k=9,
and we wrote it up as "what a well-behaved stability curve looks like."

Re-running at 50,000 abstracts broke it. Stability collapses at k=10 as expected — then
**revives to 0.93 at k=17**, and the rule, having no notion of a first stable block,
returned **18**. That is exactly the defect we had already criticised in the UMAP
measurement: a rule whose answer is set by the sweep's upper limit. The obvious repair —
stop at the first sustained collapse — returned **5** on the pilot, whose real dip at
k=6–7 reads as a cliff. Both rules were discarded.

The fix was to stop trusting curve shape and require **replication across sweeps**: a k
counts only if every independent sample found it stable. That gives 8 — the same number,
now resting on agreement between samples instead of the shape of one line.

**The lesson to state out loud:** on this corpus, any claim resting on the shape of one
curve should be assumed unreplicated until checked. That is three separate times a single
sample suggested a tidier story than the data supports — and it is why the final number is
the one all three sweeps agree on rather than the one the prettiest curve gave us.

**If asked — "so your headline number changed?"**
No. It was 8 before and it is 8 now. What changed is the argument for it, which is now
stronger: three independent samples agreeing, rather than one curve's shape.

---

## Slide 14 — Limitations and what is not done

**Headline:** What we would do next, and what we are not claiming.

**Not run:**
- **The hyperparameter search.** 10 distinct runs, specified in `tune.py` with the staged design
  below, needs a GPU. This is the clearest gap against the brief.
- **MPNet embeddings at scale** — implemented, configured, not run. Prior work suggests
  roughly +0.03 ARI.
- The 50,000-paper train-only clustering robustness check.

**Claimed carefully:**
- **k=8 is a criterion, not a truth.** k=3–8 are all reproducible.
- **The reference taxonomy is soft.** 52.7% of papers carry multiple labels, so "the"
  category of a paper often does not exist.
- **Clustering saw the whole sample.** KMeans was fitted on all 150,000, so validation and
  test abstracts contributed to defining the categories the classifier is scored on. The
  split was still fixed first, and a train-only clustering subset is reserved and verified
  for the robustness check.
- **HDBSCAN was swept on 50,000**, not 150,000 — the full set exhausted memory.
- **One encoder, one corpus slice**, 2016 onward. The taxonomy describes contemporary
  research, not the archive's history: the language-models category would not appear in a
  1990s corpus.

**The search design, if asked.** A full grid over learning rate × batch size × encoder ×
seed is 108 runs. Resolving each factor in turn and carrying the winner forward gives the
same decisions in **10**:

| Stage | Varies | Runs |
|---|---|---|
| 1 | learning rate (1e-5 … 5e-5) | 4 |
| 2 | batch size (8/16/32) | 2 |
| 3 | encoder (bert-base / DistilBERT / SciBERT) | 2 |
| 4 | seeds (42 / 1337 / 2024) | 2 |

Learning rate first because it dominates. Stage 4 is not tuning — it re-runs the winner
under three seeds so the headline carries a spread.

**Both searches are in the repo.** `tune.py --search staged` is the 10-run design;
`tune.py --search grid` is the full 108-run Cartesian product, which makes no
separability assumption and measures lr × batch interactions directly. Same runner, same
selection rule, same outputs. Neither has been run — at ~87 CPU-hours per run at full
scale, the grid is a cluster workload.

**The hyperparameters we did use:** lr 2e-5, batch 16, 3 epochs, 256 tokens — the
defaults recommended in the BERT paper (Devlin et al. 2019, Appendix A.3), and inside the
range the reference paper searched with Optuna. Say "published defaults", not "we picked
something".

**If asked — "why not just run the grid?"**
108 runs. We implemented it so the escalation is one flag, and ran one configuration to
validate the harness. With GPU budget, the order is: staged search first, then the grid
if the batch-size winner moves off 16 — which is the only place the staged design's
assumption is weak.

**If asked — "did you use the same hyperparameters as the paper?"**
Ours fall inside their search space, but the paper tunes with Optuna and **never reports
the selected values**, so nobody can match them exactly from what is published. We differ
deliberately on one axis: they truncate at 128 tokens, we use 256, because our EDA shows
256 fits 67.2% of abstracts whole against 20.5% at 128.

---

## Slide 15 — Conclusions

**Headline:** Eight categories, chosen by reproducibility, including one arXiv does not
have.

1. **Eight.** The largest k that every one of three independent sweeps — 10,000, 20,000
   and 50,000 abstracts — found reproducible, against a structureless null two orders of
   magnitude below. No arXiv label consulted.
2. **The categories mean something.** Named from distinguishing vocabulary, representative
   abstracts and archive composition; nothing degenerate.
3. **One has no arXiv equivalent** — large language models, 11.9% of the corpus. Found
   twice, independently, at two scales.
4. **They are learnable from raw text.** tf-idf + SVM 0.9005; fine-tuned BERT 0.9013 at
   pilot scale.
5. **The taxonomy and the classifier were never traded against each other.** Different
   instruments, separate verdicts.

**Close on this:** the headline number would have been the same from a much sloppier
process. What the report adds is the evidence that it is the *right* number — a null
model, a stated rule, a replication at two scales, two algorithms, and a record of the
three times we got it wrong first.

---

## Numbers cheat sheet

| | |
|---|---|
| Records scanned | 3,224,635 |
| Eligible after filters | 2,113,230 |
| Main corpus / pilot | 150,000 / 10,000 |
| Splits (main) | 105,000 / 22,499 / 22,501 |
| Multi-label / cross-group | 52.7% / 17.3% |
| Embedding | MiniLM-L6-v2, 384d, 256 tokens |
| Clustering sweep | 23 configurations (18 KMeans + 5 HDBSCAN) |
| Silhouette optimum | k=4 — 45.6% in one cluster |
| DBCV optimum | 2 clusters — 82.7% in one |
| Stability sweeps | 3 independent: n=10,000 / 20,000 / 50,000 |
| **Selected k** | **8** |
| **k=8 lower bound** | **0.922 / 0.946 / 0.893** (null 0.005–0.008) |
| Stable in every sweep | k = 3, 5, 8 |
| HDBSCAN agrees | k=7 on the pilot |
| Category shares | 7.8% – 19.2% |
| LLM category | 11.9%, 94.6% cs |
| Agreement (archives) | H 0.469 / C 0.490 |
| majority baseline | 0.0404 macro-F1 / 19.3% accuracy |
| tf-idf + SVM | 0.9005 val / 0.8970 test |
| BERT (pilot) | 0.9013 val / 0.9008 test |
| BERT training time | 349.5 min CPU |
| Full-scale projection | ~87 h/run on CPU, ~3 h on a T4 |
