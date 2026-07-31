# One-dimensional lifecycle reference

This deterministic reference isolates VRSE's lifecycle and boundary behavior from
high-dimensional representation learning and external datasets. It is an executable
contract witness, not a second performance benchmark.

## Construction

The frozen baseline is

\[
f_0(x)=0.5x.
\]

The current target adds a regional offset:

\[
y(x)=0.5x+2.5\,\mathbf 1[3\le x\le4].
\]

Five deterministic data roles are generated from disjoint grids:

| Role | Input | Size | Purpose |
|---|---|---:|---|
| baseline fit | `[-1, 1]` | 128 | establish the known service |
| calibration | `[-0.997, 0.997]` | 2,000 | calibrate known support |
| shadow observation | `[3, 4]` | 160 | learn the `+2.5` residual |
| promotion validation | `[3.01, 3.99]` | 120 | held-out utility and region evidence |
| protected guard | `[-0.95, 0.95]` | 120 | verify old-region non-overlap |

An adjacent-unknown probe uses `[5.0, 5.8]`. Although its raw interval is disjoint,
the reference should not authorize it merely because it lies near the new condition.

The example uses the `regional_regression` preset with 120 feature-training epochs,
64 minimum shadow samples and fixed random seed 7.

## Positive candidate

The candidate observes the true `+2.5` regional residual. Before review, the served
prediction remains identical to the baseline. After held-out review, the candidate
is promoted inside the observed-span region.

Expected checks:

| Property | Expected result |
|---|---:|
| maximum served change during observation | `0` |
| useful promotion | `true` |
| supported-region route fraction | `100%` |
| supported-region RMSE | approximately `2.500 → 0.003` |
| protected-region maximum baseline difference | `0` |
| adjacent-unknown maximum baseline difference | `0` |

## Harmful candidate

A separate model observes the deliberately wrong rule

\[
y_{\mathrm{harmful}}(x)=0.5x-2.5
\]

but is reviewed against the true `+2.5` target. The proposal must fail and the
candidate must not serve.

## Rollback

After the first useful promotion, `revoke()` consumes the one restore point and
returns every input to the frozen baseline. The maximum post-revoke difference from
the baseline is exactly zero in the deterministic run.

## What this reference establishes

- the public lifecycle is runnable without external data;
- shadow observation and serving use separate objects;
- useful and harmful proposal paths are both exercised;
- exact fallback holds for tested protected and unknown probes;
- observed-span authorization has deterministic boundary tests; and
- one-step rollback restores the previous snapshot.

It does not establish cross-domain generalization, high-dimensional support quality,
production reliability or a statistical risk guarantee.

## Run and inspect

```bash
python -m examples.quickstart
python -m examples.continual_stream
python -m pytest tests/test_public_contract.py tests/test_calibration.py -q
```

The data are fully defined by the formulas and grids above, so no raw data artifact is
needed. The longer exploratory analyses remain in the local research-history branch.
