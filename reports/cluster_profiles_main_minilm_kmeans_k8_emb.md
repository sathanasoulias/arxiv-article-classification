# Discovered categories — main/minilm, kmeans_k8_emb (embedding space)

8 clusters over 150,000 abstracts.

Cluster numbers below are arbitrary identifiers. Names are the author's reading of the c-TF-IDF terms and representative titles, not model output.

## Agreement with the arXiv taxonomy

Interpretation only — no arXiv label was used to select this configuration.

| reference level | homogeneity | completeness | V-measure |
|---|---|---|---|
| 20 archives | 0.4690 | 0.4900 | 0.4793 |
| 8 groups | 0.5263 | 0.3567 | 0.4252 |

## Cluster profiles

### Cluster 0 — 28,859 papers (19.2% of corpus)

**Distinguishing terms:** algorithm, problem, model, performance, algorithms, systems, optimization, network, control, framework, optimal, analysis

**Dominant arXiv archive:** `cs` (51.4% of the cluster) · **group:** cs (51.4%)

**Archives above 10%:** cs, math, eess

**Papers nearest the centroid:**

- [2207.11126](https://arxiv.org/abs/2207.11126) (`cs`) — Optimism in Face of a Context: Regret Guarantees for Stochastic Contextual MDP
- [2405.06627](https://arxiv.org/abs/2405.06627) (`cs`) — Conformal Validity Guarantees Exist for Any Data Distribution (and How to Find Them)
- [2502.05510](https://arxiv.org/abs/2502.05510) (`eess`) — Data-Driven Certificate Synthesis

### Cluster 1 — 11,697 papers (7.8% of corpus)

**Distinguishing terms:** theory, field, model, black, mass, matter, energy, scalar, gravity, gauge, dark, quantum

**Dominant arXiv archive:** `hep-ph` (29.8% of the cluster) · **group:** physics (98.9%)

**Archives above 10%:** hep-ph, hep-th, gr-qc

**Papers nearest the centroid:**

- [2110.03884](https://arxiv.org/abs/2110.03884) (`hep-ph`) — SO(10) Grand Unification with Minimal Dark Matter and Color Octet Scalars
- [2203.09002](https://arxiv.org/abs/2203.09002) (`hep-ph`) — A Natural Model of Spontaneous CP Violation
- [2301.13291](https://arxiv.org/abs/2301.13291) (`hep-th`) — Effective approach to the Antoniadis-Mottola model: quantum decoupling of the higher derivative terms

### Cluster 2 — 17,890 papers (11.9% of corpus)

**Distinguishing terms:** models, language, model, llms, reasoning, tasks, performance, large, learning, knowledge, task, framework

**Dominant arXiv archive:** `cs` (94.6% of the cluster) · **group:** cs (94.6%)

**Archives above 10%:** cs

**Papers nearest the centroid:**

- [2310.15147](https://arxiv.org/abs/2310.15147) (`cs`) — S3Eval: A Synthetic, Scalable, Systematic Evaluation Suite for Large Language Models
- [2502.21321](https://arxiv.org/abs/2502.21321) (`cs`) — LLM Post-Training: A Deep Dive into Reasoning Large Language Models
- [2309.10691](https://arxiv.org/abs/2309.10691) (`cs`) — MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback

### Cluster 3 — 16,063 papers (10.7% of corpus)

**Distinguishing terms:** equation, equations, solutions, numerical, model, boundary, problem, solution, mathbb, flow, prove, dynamics

**Dominant arXiv archive:** `math` (59.2% of the cluster) · **group:** math (59.2%)

**Archives above 10%:** math, physics

**Papers nearest the centroid:**

- [1909.08899](https://arxiv.org/abs/1909.08899) (`math`) — Finite-Volume approximation of the invariant measure of a viscous stochastic scalar conservation law
- [2408.14642](https://arxiv.org/abs/2408.14642) (`math`) — The Lake equation as a supercritical mean-field limit
- [2511.02073](https://arxiv.org/abs/2511.02073) (`math`) — Gradient bounds for viscosity solutions to certain elliptic equations

### Cluster 4 — 17,026 papers (11.4% of corpus)

**Distinguishing terms:** mathbb, mathcal, prove, group, graph, algebra, finite, graphs, groups, space, let, algebras

**Dominant arXiv archive:** `math` (85.9% of the cluster) · **group:** math (85.9%)

**Archives above 10%:** math

**Papers nearest the centroid:**

- [2312.10752](https://arxiv.org/abs/2312.10752) (`math`) — Constraints for $b$-deformed constellations
- [2204.02048](https://arxiv.org/abs/2204.02048) (`math`) — Graded Lie Algebras, Compactified Jacobians and Arithmetic Statistics
- [2111.06351](https://arxiv.org/abs/2111.06351) (`math`) — GIT stability of linear maps on projective space with marked points

### Cluster 5 — 12,640 papers (8.4% of corpus)

**Distinguishing terms:** mass, stars, galaxies, star, stellar, observations, emission, galaxy, formation, observed, gas, sim

**Dominant arXiv archive:** `astro-ph` (84.8% of the cluster) · **group:** physics (99.6%)

**Archives above 10%:** astro-ph

**Papers nearest the centroid:**

- [2307.03320](https://arxiv.org/abs/2307.03320) (`astro-ph`) — Discovering new B[e] supergiants and candidate Luminous Blue Variables in nearby galaxies
- [1808.05208](https://arxiv.org/abs/1808.05208) (`astro-ph`) — The evolution of the UV-to-mm extragalactic background light: evidence for a top-heavy initial mass function?
- [1902.06760](https://arxiv.org/abs/1902.06760) (`astro-ph`) — First Identification of 10-kpc Scale [CII] 158um Halos around Star-Forming Galaxies at z=5-7

### Cluster 6 — 20,346 papers (13.6% of corpus)

**Distinguishing terms:** quantum, states, magnetic, phase, spin, state, field, optical, systems, energy, temperature, properties

**Dominant arXiv archive:** `cond-mat` (49.6% of the cluster) · **group:** physics (97.4%)

**Archives above 10%:** cond-mat, physics, quant-ph

**Papers nearest the centroid:**

- [2207.07133](https://arxiv.org/abs/2207.07133) (`cond-mat`) — Thermodynamic characteristic for correlated flat-band system with quantum anomalous Hall ground state
- [2209.06912](https://arxiv.org/abs/2209.06912) (`cond-mat`) — Unprotected edge modes in quantum spin Hall insulator candidate materials
- [2007.15641](https://arxiv.org/abs/2007.15641) (`cond-mat`) — Moir\'{e} versus Mott: Incommensuration and Interaction in One-Dimensional Bichromatic Lattices

### Cluster 7 — 25,479 papers (17.0% of corpus)

**Distinguishing terms:** learning, model, models, image, training, neural, performance, network, deep, images, networks, datasets

**Dominant arXiv archive:** `cs` (80.4% of the cluster) · **group:** cs (80.4%)

**Archives above 10%:** cs

**Papers nearest the centroid:**

- [2610.19743](https://arxiv.org/abs/2610.19743) (`cs`) — Discriminative Score Function: Turning Pretrained Models into Functional Generative Priors
- [2103.03510](https://arxiv.org/abs/2103.03510) (`cs`) — Variational Structured Attention Networks for Deep Visual Representation Learning
- [2210.05657](https://arxiv.org/abs/2210.05657) (`cs`) — The Unreasonable Effectiveness of Fully-Connected Layers for Low-Data Regimes

## Full cluster x archive contingency table

|   row_0 |   astro-ph |   cond-mat |    cs |   econ |   eess |   gr-qc |   hep-ex |   hep-lat |   hep-ph |   hep-th |   math |   math-ph |   nlin |   nucl-ex |   nucl-th |   physics |   q-bio |   q-fin |   quant-ph |   stat |
|--------:|-----------:|-----------:|------:|-------:|-------:|--------:|---------:|----------:|---------:|---------:|-------:|----------:|-------:|----------:|----------:|----------:|--------:|--------:|-----------:|-------:|
|       0 |         92 |        232 | 14831 |    810 |   2999 |       2 |        7 |         4 |       13 |        7 |   3568 |         8 |     58 |         1 |         6 |      1414 |     797 |     496 |        669 |   2845 |
|       1 |        569 |        317 |    13 |      0 |      0 |    2100 |      557 |       375 |     3480 |     2345 |    102 |       130 |     13 |       221 |       716 |       502 |       6 |       1 |        248 |      2 |
|       2 |         17 |         30 | 16921 |     46 |    331 |       1 |        5 |         1 |        0 |        1 |     22 |         0 |      0 |         0 |         1 |        93 |     127 |      38 |          8 |    248 |
|       3 |        139 |       1580 |   305 |     33 |    151 |     115 |        1 |        17 |       20 |      122 |   9517 |       498 |    404 |         0 |        43 |      2348 |     250 |     109 |        324 |     87 |
|       4 |          4 |         69 |  1145 |     14 |      9 |      41 |        0 |         5 |        8 |      393 |  14622 |       309 |     51 |         0 |         1 |        41 |      16 |       3 |        289 |      6 |
|       5 |      10714 |         20 |    25 |      0 |     18 |     410 |       90 |         0 |      199 |        3 |      4 |         1 |      4 |       106 |        71 |       962 |       3 |       0 |          4 |      6 |
|       6 |         83 |      10084 |   215 |      1 |    103 |      20 |       10 |        29 |       82 |      114 |     88 |        98 |    102 |        26 |       113 |      4675 |     111 |       5 |       4384 |      3 |
|       7 |        140 |        172 | 20475 |     12 |   2206 |      11 |       26 |         2 |       44 |        3 |    305 |         3 |     12 |         0 |         7 |       679 |     417 |      44 |        130 |    791 |
