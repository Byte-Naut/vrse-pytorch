# Phase-3B frozen snapshot

> Status: **FROZEN — PASS**  
> Protocol: `phase3b-discovery-global-normalization-v1`  
> Frozen: 2026-07-29  
> Integrity manifest: `results/phase3b_snapshot.sha256`

Phase-3B is the terminal result of the C-MAPSS FD002 pipeline. The pipeline is
closed: future work may reproduce it, but must not tune or overwrite it.

## Result at a glance

| Claim | Frozen result |
|---|---:|
| Preconditions | READY_FOR_MATRIX, 5/5 seeds |
| Stable candidates promoted | 5/5 |
| Reversed candidates falsely promoted | 0/5 |
| Exact ID routing invariance | 5/5 |
| Exact adjacent-unknown fallback | 5/5 |
| New-regime route coverage | 93.0–96.0% |
| Frozen baseline new-regime RMSE | 96.18 mean |
| VRSE new-regime RMSE | 21.61 mean |
| RMSE reduction | 77.5% |
| Maximum normalized absolute input | 2.567 |

The stable stream shows useful adaptation rather than a vacuous rejection:
VRSE routes most supported new-regime samples to the promoted expert and reduces
RMSE by roughly 4.5x, while the protected ID and adjacent unknown regime remain on
the exact frozen service. The paired reversed-label control is rejected in every
seed. Its validation ratio reflects both a worse candidate and an easier baseline
denominator under label inversion; it is evidence of real rejection, not a claim
that only candidate degradation caused the decision.

## Interpretation boundary

This snapshot supports selective regional adaptation on one 24-dimensional
industrial simulation benchmark with one promotion event and a controlled negative
stream. It does not establish production safety, adversarial robustness, universal
OOD detection, multi-round expert composition, or state-of-the-art RUL accuracy.

## Provenance chain

1. The first precondition run exposed and preserved a spectral-norm estimator issue.
2. Phase-3A completed mechanically but was invalidated by catastrophic ID-only
   normalization; its raw artifacts remain in `results/run1_invalid_normalization/`.
3. Phase-3B changed only the pre-registered unlabeled normalization source, passed
   its hard gates, and produced the frozen result above.

The SHA-256 manifest covers the protocol, implementation, raw matrix, verdict,
checkpoints and figures needed to identify this exact result.

## Public-export redaction

The public repository replaces the local Python executable, user-home and workspace
paths embedded in `phase3_preconditions.json` with neutral placeholders. This changes
that file's byte hash but does not alter its commands, return codes, metrics or
verdict. The manifest records the hash of the redacted public copy; all other frozen
artifact hashes remain those produced by the completed experiment.
