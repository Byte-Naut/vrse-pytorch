# VRSE theory and claim boundary

This document states the abstraction implemented by the current research alpha. It
separates algebraic/software properties from empirical observations and from open
statistical questions.

## 1. Problem

Let a previously evaluated scalar-regression service be the frozen baseline

\[
f_0:\mathcal X\rightarrow\mathbb R.
\]

A labelled stream from a newly appearing condition may contain useful structure, but
it may also be temporary, mislabeled or unsupported. Updating `f₀` globally couples
two decisions:

1. learn from the stream; and
2. immediately change what every input receives.

VRSE separates them. A shadow residual candidate learns from new labels without
serving. Independent evidence can later authorize a frozen candidate for a bounded
input region.

## 2. Formal objects

| Symbol | Meaning |
|---|---|
| \(f_0\) | immutable user-supplied baseline service |
| \(e_t\) | frozen residual expert contained in deployment snapshot \(t\) |
| \(A_t\subseteq\mathcal X\) | input region authorized for snapshot \(t\) |
| \(D_{\mathrm{fit}}\) | known-condition fit data |
| \(D_{\mathrm{cal}}\) | independent known-condition calibration inputs |
| \(D_{\mathrm{obs}}\) | new-condition observations used to update the shadow |
| \(D_{\mathrm{val}}\) | held-out new-condition promotion evidence |
| \(D_{\mathrm{guard}}\) | protected known-condition inputs used to audit overlap |
| \(P_t\) | single-use proposal binding a candidate, region and configuration |

The served function is

\[
F_t(x)=
\begin{cases}
f_0(x)+e_t(x), & x\in A_t,\\
f_0(x), & x\notin A_t.
\end{cases}
\]

Before a successful promotion, \(A_t=\varnothing\), so all inputs receive `f₀`.

## 3. Data-role separation

The meaning of promotion depends on distinct roles:

| Data | Allowed use | Disallowed use |
|---|---|---|
| \(D_{\mathrm{fit}}\) | fit the frozen representation and initial residual head | promotion evidence |
| \(D_{\mathrm{cal}}\) | calibrate the known-condition support threshold | candidate training |
| \(D_{\mathrm{obs}}\) | update the isolated shadow posterior | held-out competence claim |
| \(D_{\mathrm{val}}\) | compare candidate and current service; define demonstrated support | later training followed by reuse as the same exam |
| \(D_{\mathrm{guard}}\) | test protected-region overlap and service change | training the candidate |

The implementation rejects the same tensor object for fit and calibration, but it
cannot prove that two different tensors did not contain duplicate or leaked samples.
Dataset-level independence remains the experimenter's responsibility.

## 4. Candidate and authorization region

The current expert is a Bayesian linear residual head over fixed random Fourier
features of a frozen spectrally constrained feature map. With

\[
y=f_0(x)+\psi(\phi(x))^\top w+\epsilon,
\]

the posterior is represented by sufficient statistics

\[
\Lambda=\lambda I+\sigma^{-2}\Psi^\top\Psi,
\qquad
q=\sigma^{-2}\Psi^\top r.
\]

`observe()` updates only the shadow copy of `(Λ, q)`. Prediction uses Cholesky solves
rather than an explicit matrix inverse.

The reference implementation provides two region builders:

- **Observed span:** for one-dimensional inputs, authorize the closed span supported
  by observation and validation inputs, subject to boundary and protected-range
  checks.
- **KNN feature support:** for multidimensional inputs, freeze prototypes in `φ(x)`
  space and authorize points whose kth-neighbour distance and candidate uncertainty
  fall below frozen thresholds.

Neither construction is claimed to recover semantic support in arbitrary learned
representations. They are concrete, inspectable reference rules that can be
challenged or replaced.

For stream collection, let \(u_0(x)\) and \(\tau_0\) be the frozen fit-time
uncertainty score and calibration threshold, and let \(A_t(x)\) be the current
`route_mask()`.
The public `review_mask()` is

\[
R_t(x)=[u_0(x)>\tau_0]\land\neg A_t(x).
\]

It selects inputs unfamiliar to the baseline and not already served by an active
expert. It is a collection rule, not a universal OOD statement.

## 5. Promotion rule

For the current regression implementation, a proposal can pass only when all of the
following hold:

1. candidate RMSE on \(D_{\mathrm{val}}\) meets the configured ratio relative to the
   current default service;
2. candidate 95th-percentile absolute error meets its configured ratio;
3. a non-empty authorization region can be constructed from evidence available at
   review time; and
4. the region does not route protected guard inputs to the candidate or alter their
   outputs beyond the exact-fidelity tolerance.

This is a deterministic decision conditional on the supplied datasets and
configuration. It is not a hypothesis test and currently has no multiple-review or
optional-stopping correction.

## 6. Software invariants

### Invariant A — shadow non-interference

Between promotions, `observe()` updates only the shadow head. Routing uses the frozen
deployment snapshot, never the live shadow.

**Consequence.** For any fixed input `x`, calling only `observe()` leaves the served
output unchanged.

**Proof sketch.** The forward path reads the baseline plus the deployed head and
region. `observe()` mutates a separate head and increments candidate state; none of
the forward-path objects are replaced.

### Invariant B — exact fallback

For `x ∉ Aₜ`, routing constructs a zero residual:

\[
F_t(x)=f_0(x)+0=f_0(x).
\]

This is a pointwise algebraic property of the implementation, not an average
non-forgetting claim.

### Invariant C — proposal binding and single use

`evaluate()` fingerprints the live baseline, configuration and candidate, freezes
the proposed region and candidate head, and issues a fresh token. `promote()`
recomputes the fingerprints and accepts only the latest unused token.

**Consequence.** A proposal becomes stale after candidate or configuration changes;
copying public proposal fields does not mint a valid authorization.

### Invariant D — atomic promotion

The candidate head and region are members of one immutable deployment snapshot and
are installed together.

**Consequence.** The service cannot intentionally combine a tested candidate with an
untested region through the public lifecycle.

### Invariant E — one-step rollback

Promotion stores exactly one previous deployment snapshot. `revoke()` restores that
snapshot atomically and consumes the restore point.

**Consequence.** The implementation supports one restore operation, not an unbounded
history or transactional multi-step rollback.

## 7. State transitions

```text
QUARANTINE --evaluate--> PENDING_EVAL --promote(pass)--> AUTHORIZED
     ^                         |                              |
     |                         +--promote(fail)---------------+
     |                                                        |
     +---------------------- revoke(first) -------------------+

AUTHORIZED --observe/evaluate--> AUTHORIZED
AUTHORIZED --promote(pass)------> AUTHORIZED (new snapshot)
AUTHORIZED --revoke-------------> previous snapshot
```

A failed review never silently replaces an already authorized snapshot. `REVOKED` is
an audit event, not a resting state that prevents later observation or review.

## 8. What is and is not guaranteed statistically

The known-condition uncertainty threshold uses a one-sided, distribution-free order
statistic. Under independent and identically distributed calibration inputs, it gives
the configured population-coverage statement for that uncertainty score. If the
sample is too small, calibration fails closed.

That result does **not** imply:

- the score detects every distribution shift;
- the learned feature geometry is semantically correct;
- the promoted expert has a finite-sample risk bound inside \(A_t\);
- validation remains calibrated after repeated or adaptive reviews;
- poisoned labels or adversarial inputs are rejected; or
- the baseline itself is safe.

The most important open statistical problem is to connect a promotion decision and a
data-dependent authorization region to an interpretable risk statement without
making the region uselessly small.

## 9. Empirical support

Current experiments support only the following bounded statement:

> In a deterministic one-dimensional example and one five-seed,
> 24-dimensional C-MAPSS FD002 industrial simulation study, the implemented
> lifecycle isolated shadow learning, promoted useful candidates, rejected the paired
> reversed-label candidates, routed most demonstrated new-condition inputs to the
> expert, and kept the tested old and adjacent-unknown inputs on exact fallback.

See the executable [`ONE_DIMENSIONAL_REFERENCE.md`](ONE_DIMENSIONAL_REFERENCE.md),
[`RESEARCH_SCOPE.md`](RESEARCH_SCOPE.md) for the authoritative claim map and
[`../results/CMAPSS_FD002_SNAPSHOT.md`](../results/CMAPSS_FD002_SNAPSHOT.md) for the
frozen numbers.

## 10. Research questions

- Is “validated regional permission” a useful continual-adaptation abstraction, or
  is an existing formulation more precise?
- Which support rules fail least badly across tabular, sensor, image and sequence
  representations?
- Which online simulations distinguish regional permission from replay,
  regularization or reject-all?
- How should repeated promotions and overlapping regions compose while retaining
  exact fallback?
- Which model adapters preserve practical utility without coupling the shadow to the
  baseline's mutable state?

The repository publishes a reference implementation to make these questions
testable, not to claim that they are resolved.
