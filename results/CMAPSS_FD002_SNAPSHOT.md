# C-MAPSS FD002 frozen snapshot

> Status: **FROZEN — PASS**
> Historical protocol ID: `phase3b-discovery-global-normalization-v1`
> Frozen: 2026-07-29
> Release integrity manifest: `results/cmapss_fd002_snapshot.sha256`

The historical protocol ID is intentionally retained inside the raw artifacts for
provenance. Public filenames use the benchmark and artifact purpose.

## Result at a glance

| Claim | Frozen result |
|---|---:|
| Preconditions | READY_FOR_MATRIX, 5/5 seeds |
| Stable candidates promoted | 5/5 |
| Reversed candidates falsely promoted | 0/5 |
| Exact protected-condition fallback | 5/5 |
| Exact adjacent-unknown fallback | 5/5 |
| New-regime route coverage | 93.0–96.0% |
| Frozen baseline new-regime RMSE | 96.18 mean |
| VRSE new-regime RMSE | 21.61 mean |
| RMSE reduction | 77.5% |
| Maximum normalized absolute input | 2.567 |

The stable stream shows useful adaptation rather than vacuous rejection: most
supported new-regime samples route to the promoted expert while protected and
adjacent-unknown inputs stay on exact fallback. The paired reversed-label control is
rejected in every seed.

Reversed labels change both candidate error and the baseline denominator. This is
evidence of controlled rejection, not a claim that candidate degradation alone caused
the decision.

## Interpretation boundary

This snapshot supports selective regional adaptation on one 24-dimensional industrial
simulation benchmark with one promotion event and a controlled negative stream. It
does not establish production safety, universal unknown detection, classification,
multi-round composition, adversarial robustness or state-of-the-art remaining-life
accuracy.

## Evidence chain

1. `cmapss_fd002_data_manifest.json` fixes data hashes, regime definition,
   normalization and role sizes.
2. `cmapss_fd002_preconditions.json` records integrity and learnability gates.
3. `cmapss_fd002_matrix.json` contains every method/stream/seed result.
4. `cmapss_fd002_verdict.json` mechanically evaluates the frozen criteria.
5. The metric table and figures are derived views of the same matrix.

The local research-history branch retains invalid normalization runs and earlier
exploration. They are not part of the concise public release evidence.
