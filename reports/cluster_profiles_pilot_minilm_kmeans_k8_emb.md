# Discovered categories — pilot/minilm, kmeans_k8_emb (embedding space)

8 clusters over 10,000 abstracts.

Cluster numbers below are arbitrary identifiers. Names are the author's reading of the c-TF-IDF terms and representative titles, not model output.

## Agreement with the arXiv taxonomy

Interpretation only — no arXiv label was used to select this configuration.

| reference level | homogeneity | completeness | V-measure |
|---|---|---|---|
| 20 archives | 0.4327 | 0.5165 | 0.4709 |
| 8 groups | 0.4709 | 0.3357 | 0.3920 |

## Cluster profiles

### Cluster 0 — 2,026 papers (20.3% of corpus)

**Distinguishing terms:** algorithm, problem, model, optimal, control, performance, systems, network, algorithms, information, framework, optimization

**Dominant arXiv archive:** `cs` (43.3% of the cluster) · **group:** cs (43.3%)

**Archives above 10%:** cs, math, stat

**Papers nearest the centroid:**

- [2404.00099](https://arxiv.org/abs/2404.00099) (`cs`) — Efficient and Sharp Off-Policy Evaluation in Robust Markov Decision Processes
- [2010.15234](https://arxiv.org/abs/2010.15234) (`cs`) — Linear Regression Games: Convergence Guarantees to Approximate Out-of-Distribution Solutions
- [1803.01462](https://arxiv.org/abs/1803.01462) (`cs`) — Optimal Status Updating for an Energy Harvesting Sensor with a Noisy Channel

### Cluster 1 — 1,059 papers (10.6% of corpus)

**Distinguishing terms:** equation, equations, solutions, numerical, solution, flow, model, boundary, problem, nonlinear, dynamics, function

**Dominant arXiv archive:** `math` (47.6% of the cluster) · **group:** math (47.6%)

**Archives above 10%:** math, physics

**Papers nearest the centroid:**

- [2003.13799](https://arxiv.org/abs/2003.13799) (`math`) — Knudsen Type Group for Time in $\mathbb{R}$ and Related Boltzmann Type Equations
- [2609.13414](https://arxiv.org/abs/2609.13414) (`math`) — Crossover asymptotics and a sharp confinement rate for the viscous Burgers equation
- [2608.05370](https://arxiv.org/abs/2608.05370) (`math`) — Unstable Manifolds of Stratified Euler Equations

### Cluster 2 — 998 papers (10.0% of corpus)

**Distinguishing terms:** models, language, model, llms, reasoning, tasks, performance, large, learning, framework, knowledge, training

**Dominant arXiv archive:** `cs` (92.9% of the cluster) · **group:** cs (92.9%)

**Archives above 10%:** cs

**Papers nearest the centroid:**

- [2310.15147](https://arxiv.org/abs/2310.15147) (`cs`) — S3Eval: A Synthetic, Scalable, Systematic Evaluation Suite for Large Language Models
- [2309.10691](https://arxiv.org/abs/2309.10691) (`cs`) — MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback
- [2512.07522](https://arxiv.org/abs/2512.07522) (`cs`) — LIME: Making LLM Data More Efficient with Linguistic Metadata Embeddings

### Cluster 3 — 978 papers (9.8% of corpus)

**Distinguishing terms:** stars, black, mass, galaxies, stellar, observations, star, emission, galaxy, hole, field, gravitational

**Dominant arXiv archive:** `astro-ph` (69.2% of the cluster) · **group:** physics (99.9%)

**Archives above 10%:** astro-ph, gr-qc

**Papers nearest the centroid:**

- [2401.14567](https://arxiv.org/abs/2401.14567) (`astro-ph`) — A dynamical measure of the black hole mass in a quasar 11 billion years ago
- [2107.11127](https://arxiv.org/abs/2107.11127) (`astro-ph`) — Broadband X-Ray Observations of the 2018 Outburst of the Changing-Look Active Galactic Nucleus NGC 1566
- [2105.05100](https://arxiv.org/abs/2105.05100) (`astro-ph`) — Galaxy Properties at the Faint End of the HI Mass Function

### Cluster 4 — 1,355 papers (13.6% of corpus)

**Distinguishing terms:** quantum, states, phase, state, magnetic, spin, systems, field, temperature, coupling, energy, optical

**Dominant arXiv archive:** `cond-mat` (46.5% of the cluster) · **group:** physics (97.7%)

**Archives above 10%:** cond-mat, quant-ph, physics

**Papers nearest the centroid:**

- [2103.10421](https://arxiv.org/abs/2103.10421) (`cond-mat`) — Realising the Symmetry-Protected Haldane Phase in Fermi-Hubbard Ladders
- [1811.08848](https://arxiv.org/abs/1811.08848) (`cond-mat`) — Enhanced correlations and superconductivity in weakly interacting partially flat band systems: a determinantal quantum Monte Carlo study
- [2205.05958](https://arxiv.org/abs/2205.05958) (`quant-ph`) — Beyond hard-core bosons in transmon arrays

### Cluster 5 — 835 papers (8.4% of corpus)

**Distinguishing terms:** decay, nuclear, energy, mass, collisions, model, bar, gev, qcd, neutrino, decays, measurements

**Dominant arXiv archive:** `hep-ph` (31.9% of the cluster) · **group:** physics (99.6%)

**Archives above 10%:** hep-ph, nucl-th, hep-ex, nucl-ex, hep-lat

**Papers nearest the centroid:**

- [2608.27558](https://arxiv.org/abs/2608.27558) (`hep-ph`) — KOFFEE: Kaon Observables at Future Fermilab Experimental Extensions
- [2406.03440](https://arxiv.org/abs/2406.03440) (`hep-ph`) — Analysis of experimental data on neutron decay for the possibility of the existence of the right vector boson $W_R$
- [2509.01652](https://arxiv.org/abs/2509.01652) (`hep-ph`) — Energy Correlators in Semi-Inclusive Electron-Positron Annihilation

### Cluster 6 — 1,254 papers (12.5% of corpus)

**Distinguishing terms:** mathbb, mathcal, theory, group, prove, algebra, space, groups, functions, finite, field, conjecture

**Dominant arXiv archive:** `math` (67.9% of the cluster) · **group:** math (67.9%)

**Archives above 10%:** math, hep-th

**Papers nearest the centroid:**

- [2608.18006](https://arxiv.org/abs/2608.18006) (`math`) — The obstruction to intrinsic perturbation for $A$-hypergeometric series
- [2608.30798](https://arxiv.org/abs/2608.30798) (`math`) — An Explicit Belyi Map for the Wiman Sextic and Cusp Forms for a Noncongruence Subgroup
- [2001.07506](https://arxiv.org/abs/2001.07506) (`math`) — Combinatorial Reid's recipe for consistent dimer models

### Cluster 7 — 1,495 papers (15.0% of corpus)

**Distinguishing terms:** learning, models, model, neural, image, training, network, performance, deep, networks, images, accuracy

**Dominant arXiv archive:** `cs` (73.1% of the cluster) · **group:** cs (73.1%)

**Archives above 10%:** cs, eess

**Papers nearest the centroid:**

- [2111.08449](https://arxiv.org/abs/2111.08449) (`cs`) — Complementary Ensemble Learning
- [2607.21592](https://arxiv.org/abs/2607.21592) (`cs`) — Unified Video Dense Prediction from Disjoint Data
- [2406.05612](https://arxiv.org/abs/2406.05612) (`cs`) — Which Backbone to Use: A Resource-efficient Domain Specific Comparison for Computer Vision

## Full cluster x archive contingency table

|   row_0 |   astro-ph |   cond-mat |   cs |   econ |   eess |   gr-qc |   hep-ex |   hep-lat |   hep-ph |   hep-th |   math |   math-ph |   nlin |   nucl-ex |   nucl-th |   physics |   q-bio |   q-fin |   quant-ph |   stat |
|--------:|-----------:|-----------:|-----:|-------:|-------:|--------:|---------:|----------:|---------:|---------:|-------:|----------:|-------:|----------:|----------:|----------:|--------:|--------:|-----------:|-------:|
|       0 |          6 |         15 |  877 |    139 |    200 |       0 |        1 |         0 |        0 |        1 |    224 |         1 |     16 |         0 |         0 |        84 |      90 |     104 |         44 |    224 |
|       1 |         12 |         95 |   19 |      2 |     15 |      26 |        0 |         4 |        4 |       12 |    504 |        55 |     76 |         0 |         7 |       182 |      22 |      12 |          5 |      7 |
|       2 |          0 |          1 |  927 |      5 |     28 |       0 |        1 |         0 |        0 |        0 |      2 |         0 |      0 |         0 |         0 |         2 |      14 |       5 |          0 |     13 |
|       3 |        677 |          1 |    0 |      0 |      1 |     186 |        5 |         0 |       20 |       39 |      0 |         1 |      1 |         1 |         5 |        38 |       0 |       0 |          3 |      0 |
|       4 |          5 |        630 |    7 |      0 |      6 |       5 |        0 |        21 |        3 |       23 |      9 |        16 |     23 |         2 |         8 |       256 |       8 |       0 |        332 |      1 |
|       5 |         16 |          8 |    2 |      0 |      0 |       3 |      128 |        84 |      266 |       18 |      1 |         3 |      0 |       116 |       130 |        56 |       0 |       0 |          4 |      0 |
|       6 |          1 |          8 |   52 |      3 |      1 |      24 |        0 |        13 |       10 |      166 |    852 |        79 |     17 |         0 |         0 |         3 |       1 |       1 |         23 |      0 |
|       7 |         10 |          9 | 1093 |      0 |    159 |       0 |        2 |         1 |        2 |        0 |     14 |         1 |      1 |         0 |         1 |        50 |      57 |      15 |         12 |     68 |
