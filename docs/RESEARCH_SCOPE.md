# VRSE research scope

## Research question

VRSE asks whether continual adaptation can be expressed as a deployment lifecycle:

> Keep the known service immutable, learn a candidate in isolation, review it on
> independent evidence, and grant it reversible permission only for supported inputs.

The proposed contribution is the service semantics around adaptation—not a claim to
have invented distance-aware uncertainty, local experts, reject options or shadow
models individually.

## Claim map

| Claim type | Current status | Basis |
|---|---|---|
| `observe()` cannot change served output | implementation invariant | separate live-shadow and frozen-deployment objects |
| Outside the authorized region, output equals the baseline | implementation invariant | residual is set to literal zero |
| `review_mask()` selects baseline-unfamiliar inputs outside current coverage | implementation invariant | frozen fit-time uncertainty threshold intersected with the inverse route mask |
| Proposal is tied to evaluated candidate/configuration/region | implementation invariant | fingerprints, immutable snapshot and single-use token |
| Promotion installs candidate and region together | implementation invariant | atomic snapshot replacement |
| One previous snapshot can be restored | implementation invariant | bounded restore point |
| Useful regional adaptation can occur without reject-all | initial empirical support | deterministic example and five-seed C-MAPSS study |
| Harmful candidate can be rejected | initial empirical support | deterministic harmful update and paired reversed-label control |
| Tested old and adjacent unknown regions retain fallback | initial empirical support | zero expert routing in the frozen C-MAPSS matrix |
| Works across domains, tasks and model families | open | planned benchmark and adapter work |
| Authorized-region risk is statistically bounded | open | no current finite-sample promotion theorem |
| Safe for production or high-stakes control | not claimed | no operational validation or certification |

## Current evidence

### Deterministic lifecycle example

The source-only quickstart exercises isolation, useful promotion, a harmful rejected
candidate, exact old/unknown fallback and one-step rollback. It is a mechanism check,
not a performance benchmark.

### C-MAPSS FD002 case study

The frozen study uses a 24-dimensional industrial simulation benchmark, disjoint
engine-level data roles, five fixed seeds and paired stable/reversed conditions.

- stable candidates promoted: 5/5;
- reversed candidates promoted: 0/5;
- mean stable new-condition RMSE: 96.18 baseline, 21.61 VRSE;
- supported new-condition expert routing: 93.0–96.0%;
- protected old / adjacent-unknown expert routing: 0.0% / 0.0%.

This supports the mechanism in one representative high-dimensional setting. It is not
a C-MAPSS leaderboard claim and does not establish real-aircraft performance.

## Prospective relevance, not validation

The mechanism may be relevant to industrial monitoring, sensor regression, edge
feedback loops and other systems where global online fine-tuning is too aggressive.
Those settings remain hypotheses until evaluated with domain-appropriate streams,
baselines and failure criteria.

The next evidence priority is therefore application breadth: multidimensional data
from different domains, realistic online-learning simulation, multiple drift types
and adapters for different baseline/advanced model families. See
[`BENCHMARK_PLAN.md`](BENCHMARK_PLAN.md).

## Explicit non-claims

The current release does not claim:

- a general solution to continual learning or catastrophic forgetting;
- classification, structured prediction or delayed-label support;
- correctness of KNN feature support in arbitrary representations;
- repeated-promotion or overlapping-region semantics;
- adversarial, poisoned-label or strategic-user robustness;
- production reliability, closed-loop safety or regulatory certification;
- universal out-of-distribution detection; or
- state-of-the-art task accuracy.

## Authoritative evidence

- [Frozen snapshot](../results/CMAPSS_FD002_SNAPSHOT.md)
- [Mechanical result](../results/CMAPSS_FD002_RESULT.md)
- [Per-seed metrics](../results/cmapss_fd002_metrics.md)
- [Raw matrix](../results/cmapss_fd002_matrix.json)
- [Frozen protocol](CMAPSS_FD002_PROTOCOL.md)
- [Formal definitions and invariants](THEORY.md)

When prose elsewhere conflicts with this page or the frozen artifacts, use the
narrower interpretation.
