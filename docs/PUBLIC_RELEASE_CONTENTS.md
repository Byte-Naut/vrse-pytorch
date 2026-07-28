# Public release contents

Version `0.1.0a1` is exported from the research workspace as a clean, independent
repository. It intentionally contains no earlier Git history.

Included:

- the installable `vrse` package and public lifecycle tests;
- the deterministic quickstart and C-MAPSS reproduction entry points;
- the frozen Phase-3B protocol, metrics, figures, matrix and model checkpoints;
- the two invalid Phase-3 predecessor states needed to audit protocol changes;
- the one-dimensional Stage-4C regression anchors used by the precondition gate;
- release, citation, contribution and licensing metadata.

Excluded:

- C-MAPSS source or prepared samples, which must be obtained from NASA and regenerated;
- local environments, caches and machine configuration;
- exploratory notes and superseded implementation branches that are not needed to
  understand or reproduce the public claim.

The release source tree, Python artifacts and commit tag are separate identifiers.
Verify downloaded artifacts with the published `SHA256SUMS` file rather than assuming
that files with the same version string are byte-identical.
