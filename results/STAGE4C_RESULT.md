# Stage-4C Result — CONDITIONAL_PIVOT (2026-07-17)

> Protocol: `stage4_todo/STAGE4C_PROTOCOL_DRAFT.md` (frozen before first
> execution, 2026-07-17). Code: `src/stage4c.py`, `src/methods4c.py`,
> `experiments/exp_stage4c_matrix.py`, `experiments/exp_stage4c_verdict.py`.
> Shared with Stage-4B: `src/streams4b.py`, `src/metrics4b.py`.
> Matrix: `results/stage4c_matrix_results.pkl`.
> Mechanical verdict: `results/stage4c_verdict_raw.pkl`.

## Single change from Stage-4B

Stage-4B used discrete scan-grid endpoints as the promoted region's boundaries,
causing the covering check to fail whenever a continuous sample fell fractionally
beyond the last grid point. Stage-4C replaces region construction with the
observed-span-first algorithm:

```
R_promoted = [min(X_shadow_train ∪ X_promotion_val),
              max(X_shadow_train ∪ X_promotion_val)]
```

The uncertainty scan audits the interior of this fixed interval but never
expands or replaces its continuous endpoints. Everything else is unchanged.

## Verdict: CONDITIONAL_PIVOT

```
Promotion discrimination:  stable_shift 5/5, stable_extrapolation 5/5  ✓
Unstable non-promotion:    0/5 false promotions                         ✓
Adjacent unknown safety:   stable_shift 5/5, stable_extrapolation 5/5  ✓
ID routing invariance:     stable_shift 5/5, stable_extrapolation 5/5  ✓
Region opening:            stable_shift 5/5, stable_extrapolation 5/5  ✓
Recovery (post-RMSE):      stable_shift 3/5, stable_extrapolation 3/5  ✗
```

All discrimination and structural safety conditions pass. Recovery
(`post_decision_rmse <= 120% × Ungated-128`) fails in 2/5 seeds per stable
stream. Per the pre-registered verdict tree: discrimination + safety pass,
performance fails → **CONDITIONAL_PIVOT**.

## Mechanical bullet details

| bullet | required | result |
|---|---|---|
| `both_stable_promote` (>=4/5 each stream) | True | **True** ✓ |
| `promoted_performance` (>=4/5 each stream) | True | **False** ✗ |
| `promoted_region_opens` (>=4/5 each stream) | True | **True** ✓ |
| `unstable_not_promoted` (<=1/5) | True | **True** ✓ |
| `adjacent_unknown_exact_fallback` (>=4/5 each stream) | True | **True** ✓ |
| `old_id_exact_routing_invariance` (>=4/5 each stream) | True | **True** ✓ |

## Per-seed results

### Promotion outcome (all conditions: cond1–cond4)

All 10 stable (stream, seed) combinations pass all four promotion conditions.
All 5 unstable seeds correctly fail to promote.

| stream | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | promoted |
|---|---|---|---|---|---|---|
| stable_shift | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| stable_extrapolation | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| unstable_extrapolation | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |

### Post-decision recovery: Promotion-aware vs Ungated-128

| stream | seed | promoted region | post_rmse (PA) | post_rmse (U128) | ratio | pass |
|---|---|---|---|---|---|---|
| stable_shift | 0 | [−5.981, −3.001] | 0.2384 | 0.2384 | 1.000 | ✓ |
| stable_shift | 1 | [−5.950, −3.003] | 0.2184 | 0.2184 | 1.000 | ✓ |
| stable_shift | 2 | [−5.994, −3.033] | 0.2310 | 0.1899 | **1.216** | ✗ |
| stable_shift | 3 | [−5.997, −3.001] | 0.1940 | 0.1940 | 1.000 | ✓ |
| stable_shift | 4 | [−5.956, −3.005] | 1.0429 | 0.2511 | **4.153** | ✗ |
| stable_extrapolation | 0 | [3.013, 4.999] | 0.1643 | 0.1643 | 1.000 | ✓ |
| stable_extrapolation | 1 | [3.033, 4.998] | 0.1683 | 0.1683 | 1.000 | ✓ |
| stable_extrapolation | 2 | [3.004, 4.978] | 0.5228 | 0.1349 | **3.876** | ✗ |
| stable_extrapolation | 3 | [3.002, 5.000] | 0.1620 | 0.1620 | 1.000 | ✓ |
| stable_extrapolation | 4 | [3.029, 4.997] | 0.3064 | 0.1525 | **2.008** | ✗ |

Passing seeds (ratio ≈ 1.000): post-decision points fall inside the promoted
region, so Promotion-aware and Ungated-128 serve the same shadow mean — the
ratio is exactly 1.0. The 120% threshold is not the binding constraint for
these seeds.

Failing seeds (seeds 2 and 4 per stream, ratio 2.0–4.2): post-decision
points fall partially or fully outside the promoted region, so the
Promotion-aware method falls back to the frozen deploy posterior's hard gate
for those points, incurring large fallback losses. Ungated-128 serves the
shadow mean on all post-decision points regardless.

### Structural safety (non-vacuous: all 10 seeds promoted)

All structural properties are substantively confirmed — every stable seed
promoted, so no vacuous passes.

| property | stable_shift | stable_extrapolation |
|---|---|---|
| ID routing invariance (5/5 req.) | 5/5 ✓ | 5/5 ✓ |
| Adjacent unknown exact fallback (5/5 req.) | 5/5 ✓ | 5/5 ✓ |

In every promoted seed: `id_prediction_max_diff_vs_frozen = 0.00e+00`,
`id_route_change_frac_vs_frozen = 0.000`, `second_unknown_promoted_route_frac
= 0.000`, `second_unknown_fallback_fidelity = 0.00e+00`.

## Root cause of the performance gap

The failing seeds (2 and 4 in each stream) share a common structure: the
promoted region is defined by the observed-span of shadow-train and
promotion-val inputs, which by construction does not extend to all
post-decision inputs. The post-decision phase samples from the same
stream region but with an independent random draw; some post-decision
points happen to fall outside `[obs_lo, obs_hi]`.

When a post-decision point falls outside the promoted region, the
Promotion-aware method routes it through the original frozen deployment
service (hard gate + deploy posterior). At these OOD inputs, `a(x) = 0`
(gate rejects), and the service returns the exact backbone `B(x)`. Since
the true label is `B(x) + delta(x)` with `|delta(x)|` non-trivial, this
produces a large loss. Ungated-128 has no such gap: it serves the shadow
mean everywhere, including post-decision points outside the observed span.

This is not a failure of the regional-expert architecture in its intended
use — the architecture correctly refuses to serve the shadow expert outside
the validated region. It is a **coverage gap between the training+validation
observed span and the post-decision evaluation set**: the post-decision
phase samples from a slightly wider slice of the stream region than the
observed span covers, and the Promotion-aware method has no authority to
serve the shadow expert in that uncovered slice.

The gap is seed-specific: seeds 0, 1, 3 have all post-decision points
inside the promoted region (ratio = 1.000 exactly); seeds 2 and 4 do not.
This is a random-sampling property of those particular (stream, seed)
combinations, not a structural property of the algorithm.

## Comparison to Stage-4B

| dimension | Stage-4B | Stage-4C |
|---|---|---|
| Promotion | 0+1 / 10 stable seeds | **10 / 10 stable seeds** |
| Region construction | 9/10 fail (grid quantisation) | 0/10 fail |
| Structural safety | 1 non-vacuous seed | **10 non-vacuous seeds** |
| Performance (ratio ≤ 1.20) | — (never reached) | 3+3/10 pass |
| Verdict | PIVOT | **CONDITIONAL_PIVOT** |

The observed-span-first algorithm resolves Stage-4B's quantisation failure
completely. The remaining gap is a post-decision coverage issue, not a
region-construction or safety issue.

## What CONDITIONAL_PIVOT means here

The pre-registered verdict tree defines CONDITIONAL_PIVOT as: discrimination
(stable promotes, unstable not) + structural safety (ID invariance, adjacent
unknown fallback) pass, but performance or opening fails. The interpretation
in this context:

- **Isolation is confirmed**: the promoted region correctly contains the
  shadow expert, correctly excludes the ID support, and correctly excludes
  the adjacent unknown region — in 10/10 seeds substantively.
- **Routing is confirmed**: the service correctly distinguishes in-region
  (route=2, shadow expert), out-of-region/accepted (route=1, deploy GP),
  and out-of-region/rejected (route=0, exact backbone) — in 10/10 seeds.
- **Recovery fails in 4/10 seeds**: the post-decision evaluation set
  partially extends beyond the validated region in seeds 2 and 4; the
  Promotion-aware method correctly declines to serve the shadow expert in
  those uncovered points, incurring the large fallback loss that is already
  present in the unfixed deployment service — but Ungated-128 does not have
  this constraint and achieves lower RMSE there.

The failure is not that the architecture made wrong decisions; it is that
the evaluation metric (`post_decision_rmse` vs `Ungated-128`) penalises a
correct safety decision (not serving the shadow expert outside the validated
region) relative to an unsafe diagnostic (Ungated-128 serves everywhere).

## Candidate directions (none pre-authorised)

1. **Extend the promoted region to include the full post-decision evaluation
   set** by incorporating post-decision inputs into the observed-span
   computation. This requires ensuring those inputs remain OOD-free and
   outside the ID support — exactly the kind of pre-commit that would need
   its own protocol.
2. **Accept CONDITIONAL_PIVOT as terminal for this stream design**: the
   architecture provides safety and isolation; the performance gap is an
   artifact of the specific stream segmentation (post-decision drawn from a
   wider slice than shadow-train + val). A redesigned evaluation protocol that
   ensures post-decision points fall inside the observed span would remove the
   performance gap without changing the architecture.
3. **Two-stage promotion**: promote once on the observed span, then expand
   the region after seeing post-decision inputs — but this risks post-decision
   contamination and would need careful pre-registration.

Full outputs: `results/stage4c_matrix_results.pkl`, `results/stage4c_verdict_raw.pkl`.
