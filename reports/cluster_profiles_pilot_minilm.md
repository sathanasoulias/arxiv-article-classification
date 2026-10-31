# Discovered categories — pilot/minilm, hdbscan_mcs100

7 clusters over 10,000 abstracts, 11.0% left unassigned as noise.

Cluster numbers below are arbitrary identifiers. Names are the author's reading of the c-TF-IDF terms and representative titles, not model output.

## Agreement with the arXiv taxonomy

Interpretation only — no arXiv label was used to select this configuration.

| reference level | homogeneity | completeness | V-measure |
|---|---|---|---|
| 20 archives | 0.4883 | 0.7512 | 0.5919 |
| 8 groups | 0.4990 | 0.4744 | 0.4864 |

## Cluster profiles

### Cluster 0 — 4,307 papers (43.1% of corpus)

**Distinguishing terms:** model, models, learning, performance, framework, network, training, problem, information, algorithm, existing, neural

**Dominant arXiv archive:** `cs` (65.6% of the cluster) · **group:** cs (65.6%)

**Archives above 10%:** cs

**Papers nearest the centroid:**

- [1907.08392](https://arxiv.org/abs/1907.08392) (`cs`) — Automated Machine Learning in Practice: State of the Art and Recent Results
- [2006.00887](https://arxiv.org/abs/2006.00887) (`cs`) — Insights into Performance Fitness and Error Metrics for Machine Learning
- [2410.17238](https://arxiv.org/abs/2410.17238) (`cs`) — SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning

### Cluster 1 — 1,334 papers (13.3% of corpus)

**Distinguishing terms:** quantum, states, state, phase, magnetic, spin, systems, field, temperature, optical, coupling, energy

**Dominant arXiv archive:** `cond-mat` (43.5% of the cluster) · **group:** physics (95.6%)

**Archives above 10%:** cond-mat, quant-ph, physics

**Papers nearest the centroid:**

- [1912.07053](https://arxiv.org/abs/1912.07053) (`cond-mat`) — Geometric phase of Wannier-Stark ladders in alkaline-earth(-like) atoms
- [2204.07088](https://arxiv.org/abs/2204.07088) (`physics`) — Superoscillations Made Super Simple
- [1902.10883](https://arxiv.org/abs/1902.10883) (`quant-ph`) — Topologically Protected Quantum Coherence in a Superatom

### Cluster 2 — 803 papers (8.0% of corpus)

**Distinguishing terms:** decay, mass, nuclear, energy, collisions, bar, gev, model, qcd, decays, neutrino, matter

**Dominant arXiv archive:** `hep-ph` (33.4% of the cluster) · **group:** physics (100.0%)

**Archives above 10%:** hep-ph, hep-ex, nucl-th, nucl-ex

**Papers nearest the centroid:**

- [2510.24759](https://arxiv.org/abs/2510.24759) (`physics`) — Estimation of the Inverse Compton Scattering Background in MeV Gamma-Gamma Collider
- [1903.05529](https://arxiv.org/abs/1903.05529) (`nucl-ex`) — High Precision Measurement of Compton Scattering in the 5 GeV region
- [2505.07994](https://arxiv.org/abs/2505.07994) (`nucl-ex`) — Measurement of the Total Compton Scattering Cross Section between 6.5 and 11 GeV

### Cluster 3 — 658 papers (6.6% of corpus)

**Distinguishing terms:** galaxies, stars, stellar, mass, observations, galaxy, emission, star, gas, magnetic, formation, observed

**Dominant arXiv archive:** `astro-ph` (92.7% of the cluster) · **group:** physics (99.8%)

**Archives above 10%:** astro-ph

**Papers nearest the centroid:**

- [1804.07486](https://arxiv.org/abs/1804.07486) (`astro-ph`) — The Contribution of Stellar Winds to Cosmic Ray Production
- [2007.12583](https://arxiv.org/abs/2007.12583) (`astro-ph`) — The long duration cryogenic system of the OLIMPO balloon--borne experiment: design and in--flight performance
- [2312.04041](https://arxiv.org/abs/2312.04041) (`astro-ph`) — Astronomy as a Field: A Guide for Aspiring Astrophysicists

### Cluster 4 — 424 papers (4.2% of corpus)

**Distinguishing terms:** black, gravity, hole, field, theory, gravitational, quantum, scalar, spacetime, holes, gauge, energy

**Dominant arXiv archive:** `gr-qc` (42.0% of the cluster) · **group:** physics (97.4%)

**Archives above 10%:** gr-qc, hep-th

**Papers nearest the centroid:**

- [2408.07329](https://arxiv.org/abs/2408.07329) (`hep-th`) — Gravitational wave forms for extreme mass ratio collisions from supersymmetric gauge theories
- [2404.13917](https://arxiv.org/abs/2404.13917) (`hep-th`) — Decay of Kaluza-Klein Vacuum via Singular Instanton
- [1804.09177](https://arxiv.org/abs/1804.09177) (`gr-qc`) — Unruh effect as a result of quantization of spacetime

### Cluster 5 — 1,181 papers (11.8% of corpus)

**Distinguishing terms:** mathbb, prove, mathcal, group, problem, groups, space, let, graph, equation, finite, class

**Dominant arXiv archive:** `math` (89.6% of the cluster) · **group:** math (89.6%)

**Archives above 10%:** math

**Papers nearest the centroid:**

- [2311.18454](https://arxiv.org/abs/2311.18454) (`math`) — On $k$-free numbers in cyclotomic fields: entropy, symmetries and topological invariants
- [2206.09163](https://arxiv.org/abs/2206.09163) (`math`) — Asymmetric tropical distances and power diagrams
- [2207.13139](https://arxiv.org/abs/2207.13139) (`math`) — Asymptotic Betti numbers for hard squares in the homological liquid regime

### Cluster 6 — 193 papers (1.9% of corpus)

**Distinguishing terms:** flow, model, dynamics, surface, water, fluid, numerical, simulations, active, cell, liquid, molecular

**Dominant arXiv archive:** `cond-mat` (43.0% of the cluster) · **group:** physics (86.5%)

**Archives above 10%:** cond-mat, physics

**Papers nearest the centroid:**

- [2210.03384](https://arxiv.org/abs/2210.03384) (`cond-mat`) — Ultrafast reversible self-assembly of living tangled matter
- [1907.06825](https://arxiv.org/abs/1907.06825) (`cond-mat`) — Crack propagation under static and dynamic boundary conditions
- [2212.12709](https://arxiv.org/abs/2212.12709) (`physics`) — On the importance of 3D stress state in 2D earthquake rupture simulations with off-fault deformation

## Full cluster x archive contingency table

|   row_0 |   astro-ph |   cond-mat |   cs |   econ |   eess |   gr-qc |   hep-ex |   hep-lat |   hep-ph |   hep-th |   math |   math-ph |   nlin |   nucl-ex |   nucl-th |   physics |   q-bio |   q-fin |   quant-ph |   stat |
|--------:|-----------:|-----------:|-----:|-------:|-------:|--------:|---------:|----------:|---------:|---------:|-------:|----------:|-------:|----------:|----------:|----------:|--------:|--------:|-----------:|-------:|
|       0 |         11 |         22 | 2827 |    146 |    365 |       0 |        2 |         1 |        2 |        1 |    230 |         1 |     15 |         0 |         0 |       116 |     130 |     135 |          3 |    300 |
|       1 |          1 |        580 |   22 |      0 |     13 |       0 |        0 |         6 |        1 |       12 |     18 |        20 |      5 |         1 |        13 |       241 |       4 |       0 |        395 |      2 |
|       2 |         18 |          1 |    0 |      0 |      0 |       1 |      132 |        79 |      268 |       12 |      0 |         2 |      0 |       115 |       123 |        48 |       0 |       0 |          4 |      0 |
|       3 |        610 |          1 |    0 |      0 |      1 |      11 |        1 |         0 |        2 |        0 |      0 |         2 |      0 |         1 |         0 |        28 |       0 |       0 |          1 |      0 |
|       4 |          8 |          1 |    0 |      0 |      0 |     178 |        0 |         2 |       10 |      174 |     11 |        15 |      2 |         1 |         1 |        15 |       0 |       0 |          6 |      0 |
|       5 |          0 |          7 |   56 |      0 |      1 |       1 |        0 |         1 |        0 |        9 |   1058 |        37 |      5 |         0 |         0 |         0 |       2 |       1 |          1 |      2 |
|       6 |          0 |         83 |    3 |      0 |      2 |       0 |        0 |         0 |        0 |        0 |     10 |         3 |      6 |         0 |         0 |        75 |      11 |       0 |          0 |      0 |
