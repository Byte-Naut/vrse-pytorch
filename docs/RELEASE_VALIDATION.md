# Release-surface validation

> Date: 2026-07-29  
> Candidate: `0.1.0a1`

## Deterministic quickstart

Command:

```text
python -m examples.quickstart
```

Observed:

```text
VRSE quickstart
  isolated learning max output change : 0.000e+00
  promotion passed                    : True
  new-region route fraction           : 1.000
  new-region RMSE before -> after      : 2.500 -> 0.003
  old-region max fallback difference  : 0.000e+00
  unknown max fallback difference     : 0.000e+00
```

## Automated tests

```text
31 passed in 14.98s
```

The same clean public tree also passed both frozen Stage-4C anchors: 0 decision
mismatches across the 15 reference streams and exact `0.000e+00` GP state/prediction
differences.

## Release artifacts

The final source distribution and universal wheel are built directly from the clean
public tree. Their immutable SHA-256 identifiers and clean-install smoke-test record
are distributed separately as:

```text
dist/SHA256SUMS
dist/VALIDATION.txt
```

The wheel is intentionally limited to the five `vrse` Python modules, Apache-2.0
license/notice files and standard distribution metadata. The sdist additionally
contains the curated reproduction code and frozen evidence listed in
`docs/PUBLIC_RELEASE_CONTENTS.md`; it does not contain C-MAPSS samples.
