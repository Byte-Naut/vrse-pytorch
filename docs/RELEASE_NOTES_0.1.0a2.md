# VRSE 0.1.0a2 — research alpha release notes

This alpha publishes a runnable reference implementation of a conservative
continual-learning lifecycle: isolated shadow learning, held-out promotion
review, regional deployment permission, exact frozen-baseline fallback, and
one-step rollback.

## What a user can do

Wrap a scalar-regression PyTorch baseline, establish disjoint data roles,
select uncovered stream inputs with `review_mask()`, send labelled samples to
an isolated residual candidate, evaluate with
held-out validation and a protected guard, and atomically authorize the
tested snapshot inside a one-dimensional span or multidimensional KNN feature
region.

```text
wrap → fit → review_mask → observe → evaluate → promote → route / revoke
```

## Evidence included

- Deterministic no-download quickstart: useful promotion, harmful rejection,
  exact fallback, and rollback.
- Automated lifecycle, calibration, and high-dimensional support tests.
- Frozen five-seed C-MAPSS FD002 mechanism study:
  - Stable candidates promoted **5/5**; reversed-label candidates promoted
    **0/5**.
  - Mean stable new-regime RMSE: 96.18 → **21.61** (77.5% reduction).
  - Expert routing on new-regime inputs: **93.0–96.0%**.
  - Expert routing on protected old / adjacent-unknown inputs: **0.0%**.

## Interpretation boundary

This is initial mechanism evidence in one representative industrial simulation.
It does not certify baseline safety, prove universal unknown detection,
establish classification or multi-expert behavior, solve continual learning
generally, or claim state-of-the-art C-MAPSS performance.

The reversed-label control changes both candidate error and the baseline
denominator — it validates a rejection path without isolating a single causal
term.

## Positioning

VRSE is not a general continual-learning framework or production safety layer.
It is released to test the core abstraction, support independent reproduction,
and invite collaboration on broader applications and model adapters.

The package is `vrse-pytorch`, version `0.1.0a2`, under Apache-2.0. Repository:
`https://github.com/Byte-Naut/vrse-pytorch`.
