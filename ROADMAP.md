# Roadmap

VRSE is a research alpha. Priorities favor independent reproduction, realistic
applications, diverse data — over deeper theory or feature breadth.

## Current state (v0.1.0a2)

- Supervised scalar regression with one regional residual expert.
- Isolated observation, held-out promotion, exact fallback, one-step rollback.
- One 24-dimensional C-MAPSS FD002 case study across five seeds.

This release shows the lifecycle runs and produces a non-vacuous regional
adaptation. It is not evidence of general continual-learning performance.

## Near-term directions

### Automated pipeline prototype

Store uncovered stream inputs, join delayed labels by sample ID, and run
`observe → evaluate → promote` automatically.

### Broader application domains

Extend beyond the single industrial simulation to datasets with meaningfully
different geometry — wearable sensing, energy forecasting, environmental
monitoring, or image classification. The goal is to learn where regional
permission helps and where it breaks, not to accumulate benchmarks.

### Online stream simulation

Evaluate under realistic arrival patterns: abrupt shifts, gradual drift,
recurring conditions, temporary excursions, and noisy or delayed labels. A
shared stream harness with fixed label budgets makes comparisons interpretable.

### Model and task adapters

Test with MLPs, convolutional, recurrent, and transformer-style backbones —
both trained from scratch and pretrained. Add a minimal classification adapter
to learn whether promotion metrics transfer beyond regression.

### Test diverse backbone architectures

Wrap at least two materially different model families as VRSE baselines. Report
where the adapter contract works and where it fails — negative results with
clear provenance are valuable.

## Prioritization

Prefer work that exposes a new failure mode, domain, or model family over work
that only increases confidence on the existing example.

### Multiple experts

Combine promoted regions into one coverage mask and handle overlaps explicitly.
