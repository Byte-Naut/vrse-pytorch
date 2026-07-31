# Benchmark plan

The C-MAPSS result is an initial mechanism study. The next step is to test
whether regional permission remains useful across domains, data geometries,
stream dynamics, and model families — and to document where it fails.

## Core evaluation questions

Every benchmark should answer:

1. **Utility:** does the supported new condition improve after promotion?
2. **Interference:** what changes on protected old conditions?
3. **Coverage:** how often does the expert serve on demonstrated new inputs?
4. **Unknown behavior:** what happens on adjacent or novel conditions?
5. **Decision quality:** which useful candidates are rejected, and which harmful
   ones are promoted?

Accuracy without route coverage is insufficient — a method that rejects every
input can look non-interfering while providing no adaptation.

## Stream scenarios

At minimum, include several of:

- Abrupt appearance of a stable new condition.
- Gradual drift with a controllable transition.
- Recurring or returning conditions.
- Temporary shifts that later disappear.
- Label noise or delayed feedback.
- A controlled harmful update (e.g., label reversal).

Freeze stream construction, label budget, and review schedule before comparing
methods.

## Domain and data diversity

The first expansion should target domains with materially different geometry
from C-MAPSS, not nearby industrial datasets that exercise the same assumptions.
Candidate areas include wearable or activity sensing, energy or demand
forecasting, environmental monitoring, and image classification.

Different data modalities (tabular, temporal, image, pretrained embeddings) and
task types (regression, classification) should be represented.

## Model and adapter diversity

Test across a range of backbones — shallow MLPs, deeper residual networks, 1-D
convolutional or recurrent encoders for sequences, vision models for image
tasks, and transformer-style encoders. Include both models trained from scratch
and pretrained feature extractors.

An adapter that cannot satisfy the baseline contract (deterministic output,
frozen state, serialization) should fail explicitly — documenting these limits
is as valuable as reporting successes.

## Comparison methods

Use identical data access, label budgets, and review points for:

- Frozen baseline (no adaptation).
- Global online fine-tuning.
- Isolated shadow served globally after review.
- Replay and a representative regularization method.
- Reject-all / static baseline.
- VRSE regional permission.

Additional methods are welcome; comparisons without matched budgets should be
reported separately.

## Metrics

Report at least:

- **Predictive:** RMSE/MAE for regression, accuracy and a proper scoring rule
  for classification, over stream time and on the supported subset.
- **Continual:** old-condition forgetting, interference on protected probes,
  forward transfer.
- **Authorization:** expert coverage by condition, false-authorization rate,
  promotion/rejection outcomes, behavior as support radius varies.
- **Operations:** latency, memory, labels consumed, serialization fidelity.

Report distributions across fixed seeds, not only means. Publish the raw
per-seed data used to produce summary figures.

## Data discipline

- Split by the highest-level independent entity (engine, subject, site,
  sequence), not by rows.
- Keep fit, calibration, observation, validation, guard, and post-decision
  roles disjoint.
- Freeze seeds, budgets, and metrics before the comparison run.
- Preserve negative results — they are as informative as successes.

## Minimum publishable expansion

A convincing next evidence release should include: two additional domains, at
least one classification task, several drift types, multiple model families,
matched baselines, core metrics, and at least one documented case where VRSE
fails or is not the preferred method.

Breadth across domains and models is currently more valuable than a deep
reliability theorem for a single support rule whose cross-domain behavior is
still unknown.
