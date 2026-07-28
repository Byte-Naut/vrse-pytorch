# VRSE research scope

## The question

VRSE asks a narrow systems question:

> Can an online service learn a useful new regional behavior without allowing that
> learning process to modify the protected service elsewhere?

The proposed answer is a lifecycle rather than a new uncertainty score: freeze the
old service, train a candidate in isolation, test it on independent evidence, bind a
passing snapshot to a supported input region, and use exact fallback outside that
region.

## What is supported today

The implementation gives a direct routing invariant. If an input is not inside the
authorized region, the deployed output is the frozen baseline output exactly. Shadow
updates cannot alter the served snapshot, and promotion binds the evaluated candidate
and region atomically.

Two experiment families support the implementation:

- The one-dimensional Stage-4C reference isolates boundary and lifecycle behavior.
- The frozen Phase-3B study extends the mechanism to a real 24-dimensional industrial
  simulation benchmark. Across five seeds, stable candidates promote 5/5 times,
  reversed candidates promote 0/5 times, new-regime RMSE falls from 96.18 to 21.61 on
  average, and ID/adjacent-unknown routing remains exactly zero.

This is evidence that regional permission can create a useful safety–plasticity
trade-off in the tested setting. It is stronger than a toy demonstration, but still a
single-task, single-promotion result.

## What remains open

Current evidence does not establish:

- a distribution-free bound on the risk of an authorized expert;
- reliable support regions in arbitrary learned representations;
- composition, conflict resolution or rollback across multiple experts and rounds;
- robustness to poisoned labels, adversarial inputs or strategic users;
- classification, structured prediction, delayed labels or closed-loop control safety;
- production reliability or state-of-the-art task performance.

Finite evidence also creates an unavoidable coverage boundary: opening beyond observed
support may include a neighboring unknown regime, while refusing to do so may leave a
stable tail uncovered. VRSE chooses explicit, inspectable permission over pretending
that this ambiguity has disappeared.

## Academic position

The individual ingredients—distance-aware uncertainty, reject options, shadow models,
local experts and runtime fallback—are established ideas. VRSE's proposed contribution
is their deployment semantics:

> online adaptation as validated, regional and reversible permission, with exact
> non-interference outside the permitted region.

That places the project between selective prediction, continual/open-world learning,
mixtures of experts and runtime assurance. The durable research questions are how to
attach finite-sample risk guarantees to promotion and how to compose multiple regional
permissions without losing the fallback invariant.

## Evidence map

- [Frozen Phase-3B snapshot](../results/PHASE3B_SNAPSHOT.md)
- [Phase-3 protocol](Phase3_Plan.md)
- [Phase-3B amendment](Phase3B_Amendment.md)
- [Mechanical result](../results/PHASE3_RESULT.md)

The scope above and the frozen snapshot are authoritative for the public release.
