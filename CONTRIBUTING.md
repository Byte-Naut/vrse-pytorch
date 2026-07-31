# Contributing

VRSE is an early research alpha. Contributions are most useful when they make a
claim more reproducible, expose a failure mode, or test the abstraction in a
meaningfully different setting.

## Contribution tracks

### Reproduce

Run the quickstart or frozen C-MAPSS study on a clean system. Report exact
commands, versions, source-data hashes, and any deviations from the frozen
artifact. A failed reproduction is as valuable as a successful one — don't tune
the protocol until it matches.

### Challenge

Submit a minimal counterexample to the authorization rule: the smallest stream
that triggers a wrong route, promotion, or fallback decision. State which
assumption appears to fail and whether the failure is deterministic across
seeds. Boundary leakage, coverage collapse, and representation aliasing are
especially relevant.

### Extend

Add classification, a multidomain stream, a model adapter, repeated promotion,
or multiple experts. Start from a work package in
[`ROADMAP.md`](ROADMAP.md) and define acceptance criteria before implementing.

### Compare

Benchmark VRSE against global fine-tuning, replay, regularization, reject-all,
or an established CL toolkit under identical stream, label, review, and compute
budgets. Report utility, interference, coverage, and operational cost — not only
end-task accuracy.

## Claim discipline

Keep these categories separate:

- **Implementation invariant:** follows from state/routing logic, covered by a
  deterministic test.
- **Empirical observation:** supported by a named, frozen artifact.
- **Hypothesis or intended application:** proposed for future evaluation.
- **Open guarantee:** not currently established.

Don't describe exact fallback as a safety certificate, or one benchmark result
as general continual-learning validation.

## Data-role discipline

Fit, calibration, observation, validation, guard, and post-decision roles must
be disjoint. Split by the highest-level independent entity available. Document
any unavoidable role reuse explicitly.

## Development checks

```bash
python -m pip install -e ".[benchmark,test]"
python -m pytest -q
python -m examples.quickstart
python -m examples.continual_stream
```

Run C-MAPSS into a fresh output directory, not the checked-in `results/`. See
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Change requirements

1. Add or update deterministic tests for behavioral changes.
2. Keep public API changes separate from benchmark details.
3. Include raw machine-readable results behind any new table or figure.
4. Fix seeds, budgets, and stop criteria before a comparison run.
5. Update `RESEARCH_SCOPE.md` when evidence or non-claims change.
6. Don't commit datasets, prepared arrays, checkpoints, virtual environments,
   or build artifacts.

## Pull request structure

```text
Research question or failure:
Current assumption:
Change:
Acceptance criteria:
Data roles and provenance:
Results, including negative outcomes:
Claim category affected:
Remaining limitations:
```

Small, auditable contributions are preferred. A focused counterexample or clean
reproduction is often more valuable than a broad rewrite.

By submitting, you agree that your contribution may be distributed under the
[Apache License 2.0](LICENSE). Don't submit material you don't have the right to
license on those terms.
