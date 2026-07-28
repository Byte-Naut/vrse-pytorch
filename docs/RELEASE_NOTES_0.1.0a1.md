# VRSE 0.1.0a1 — candidate release notes

This alpha introduces a compact PyTorch implementation of **Validated Regional
Support Expansion**: isolated shadow learning, independent promotion validation,
regional deployment permission and exact frozen fallback.

## What a user can do

Wrap a scalar-regression baseline, fit the frozen ID service, stream labelled samples
to an isolated shadow GP, evaluate the candidate on held-out data with an ID guard,
and atomically authorize the tested snapshot inside either a one-dimensional span or a
high-dimensional KNN feature region.

The public lifecycle is:

```text
wrap → fit → observe → evaluate → promote → route / revoke
```

## Evidence included

- 31 automated lifecycle, calibration and high-dimensional tests.
- Two exact Stage-4C regression anchors retained by the Phase-3 precondition gate.
- A deterministic no-download quickstart.
- A frozen five-seed NASA C-MAPSS FD002 mechanism study:
  - stable candidates promoted 5/5;
  - reversed-label candidates promoted 0/5;
  - average stable new-regime RMSE: 96.18 → 21.61;
  - expert routing on new-regime inputs: 93.0–96.0%;
  - expert routing on protected ID and adjacent unknown inputs: 0%.

## Important interpretation

This release demonstrates a deployment contract and one representative use case. It
does not certify that the fallback is safe, prove universal OOD detection, solve
continual learning in general or claim C-MAPSS state-of-the-art performance.

The reversed-label control is intentionally synthetic. Its rejection reflects both a
worse candidate and a changed baseline-error denominator under inversion; it validates
the rejection path without identifying a single causal term.

## Reproducibility and history

Phase-3B is frozen under `phase3b-discovery-global-normalization-v1`. The snapshot and
hash manifest are in `results/PHASE3B_SNAPSHOT.md` and
`results/phase3b_snapshot.sha256`. The repository also retains two invalid historical
states: a spectral-normalization estimator failure and a Phase-3A normalization
protocol failure. Neither is presented as VRSE performance evidence.

## Release identity

This release is distributed as `vrse-pytorch` under Apache-2.0. Its canonical source
repository is `https://github.com/Byte-Naut/vrse-pytorch`. The release artifacts and
source tag are identified by SHA-256 values recorded with the release.
