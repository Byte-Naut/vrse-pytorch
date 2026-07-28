# Contributing

VRSE is currently an alpha research prototype. Small, auditable changes are preferred
over broad framework expansion.

## Good first contributions

- Reproduce the deterministic quickstart or frozen Phase-3B protocol on a clean system.
- Improve documentation without changing the frozen experimental claims.
- Add tests for lifecycle invariants, serialization or device/dtype behavior.
- Report a minimal failure case with inputs, expected behavior and environment details.

## Before opening a change

1. Keep the frozen Phase-3B pipeline and artifacts unchanged.
2. Add or update tests for any behavioral change.
3. Separate public API proposals from internal experiment implementations.
4. Do not describe routing invariants as statistical or domain safety guarantees.
5. Do not commit C-MAPSS data, generated checkpoints or pickle files.

Useful checks:

```bash
python -m pytest -q
python -m examples.quickstart
python -m experiments.exp_stage4c_vrse_regression
python -m experiments.exp_stage4c_vrse_gp_equivalence
```

By submitting a contribution, you agree that it may be distributed under the
project's [Apache License 2.0](LICENSE). Do not submit material that you do not have
the right to license on those terms.
