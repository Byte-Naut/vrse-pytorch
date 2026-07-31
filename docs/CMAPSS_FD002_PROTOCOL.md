# Frozen C-MAPSS FD002 protocol

> Status: frozen reference experiment
> Verdict: PASS
> Seeds: 4300–4304
> Historical artifact identifier: `phase3b-discovery-global-normalization-v1`

The historical identifier is retained inside the frozen JSON and checkpoints so that
incompatible prepared data is rejected. Public filenames use the benchmark's actual
purpose rather than the repository's former development sequence.

## Question

Can the VRSE lifecycle learn a useful behavior for a newly appearing operating
regime, authorize it with non-vacuous coverage, and leave a protected old regime and
an adjacent unknown regime on exact fallback?

This is a mechanism study, not a remaining-life leaderboard experiment.

## Dataset and target

- Dataset: NASA C-MAPSS FD002 training split.
- Units: 260 simulated engines.
- Input: three operational settings plus 21 sensor values (24 dimensions).
- Target: remaining cycles within each engine, capped at 125.
- Raw data: not redistributed by this repository.

The data loader requires a 26-column finite matrix and records source-file hashes in
the checked-in manifest.

## Regime discovery and normalization

Units 1–20 are reserved before any supervised split. Only their three operational
settings are used to fit a six-cluster K-means definition with fixed random state
31415 and 20 initializations. Cluster labels are made deterministic by sorting their
centers in original operational units.

- the most frequent cluster is the known regime;
- the most distant cluster center is the new regime;
- the second most distant cluster center is the adjacent-unknown probe.

The same reserved units, across all regimes, define one unlabeled mean and standard
deviation for the 24 model inputs. They never enter fit, calibration, observation,
validation, guard or post-decision roles. The frozen normalization audit requires all
values to remain finite and rejects a standard-deviation floor that hides variation
elsewhere in FD002.

## Disjoint engine roles

For each seed, units 21–260 are shuffled once and assigned as whole engines:

| Role | Engines | Use |
|---|---:|---|
| baseline fit | 60 | train the baseline and fit-time residual service |
| known calibration | 30 | calibrate the frozen support threshold |
| known guard | 30 | protected-region audit |
| shadow observation | 50 | update the isolated candidate |
| promotion validation | 35 | held-out competence and support evidence |
| post-decision evaluation | 35 | new and adjacent-unknown probes |

Rows are selected only after engine assignment and are ordered by cycle, then engine
identifier. No engine appears in two supervised roles.

## Model and candidate

Each seed trains a two-hidden-layer, 64-unit MLP baseline on the known regime. VRSE
wraps the baseline using the `regional_regression_highdim` preset. The observation
stream updates a shadow Bayesian residual head; it does not change the served
snapshot.

The multidimensional authorization region is a frozen KNN support set in the learned
feature space, intersected with a frozen candidate-uncertainty threshold. Guard
samples must have zero expert routing.

## Paired conditions

- **Stable condition:** observation, validation and post-decision labels retain the
  original capped remaining-life target.
- **Reversed condition:** inputs and observation are unchanged, while validation and
  post-decision labels are transformed as `125 - y`. This creates a controlled
  candidate that should not be authorized.

Reversal changes both candidate error and the baseline denominator. It tests whether
the lifecycle performs a real rejection; it does not identify candidate degradation
as the sole causal reason for rejection.

## Compared methods

| Method | Serving behavior |
|---|---|
| Frozen | baseline only |
| Online-ungated | live residual candidate served globally |
| Static-reject | fit-time residual with frozen uncertainty gate; no new learning |
| Shadow-global | reviewed candidate served globally if competence checks pass |
| VRSE-KNN | reviewed candidate served only inside the frozen KNN region |

Methods share the baseline, representation and data budgets within each seed.

## Preconditions

The matrix is valid only if the public contract tests and quickstart pass, data roles
are disjoint and finite, normalization passes its structural audit, spectral norms
meet their configured limits, batch and incremental posterior updates agree, and at
least four of five seeds satisfy the baseline/new-regime learnability gates.

The checked-in precondition artifact records the exact frozen run. A debug run that
skips contract gates cannot unlock a formal matrix.

## Decision criteria

The mechanical verdict evaluates:

- isolation for both streams in every seed;
- useful promotion in at least four of five stable runs;
- at most one false promotion in the reversed runs;
- exact protected and adjacent-unknown fallback plus at least 80% new-regime coverage
  in at least four stable runs; and
- stable VRSE utility relative to frozen and global-shadow comparisons in at least
  four seeds.

Criteria and seeds are read from code and frozen artifacts; they must not be retuned
when reproducing the reference result.

## Evidence files

- `results/cmapss_fd002_data_manifest.json`
- `results/cmapss_fd002_preconditions.json`
- `results/cmapss_fd002_matrix.json`
- `results/cmapss_fd002_verdict.json`
- `results/cmapss_fd002_metrics.md`
- `results/CMAPSS_FD002_RESULT.md`
- `results/CMAPSS_FD002_SNAPSHOT.md`
- three `results/cmapss_fd002_*.png` figures

See [`REPRODUCTION.md`](REPRODUCTION.md) for commands and output isolation.
