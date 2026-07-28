# VRSE 0.1.0a1 release checklist

The algorithm and Phase-3B evidence are frozen. This checklist covers only the public
release surface.

## Release identity

- [x] Select and add Apache-2.0 as the open-source `LICENSE`.
- [x] Select `vrse-pytorch` (import `vrse`) and reserve the canonical repository URL.
- [x] Use `Byte-Naut` as the canonical public creator and the hosting account's
  no-reply identity for commits. The original `v0.1.0a1` package archives retain
  their pre-DOI embedded build metadata because the frozen files were not rebuilt.

## Package

- [x] Add PEP 517/621 package metadata in `pyproject.toml`.
- [x] Build and inspect a candidate wheel (`vrse_pytorch-0.1.0a1-py3-none-any.whl`).
- [x] Build the final wheel and source distribution from the clean public tree.
- [x] Install the candidate wheel into a second clean environment and verify public imports.
- [x] Install the final wheel and the wheel rebuilt from the final sdist in two clean environments.
- [x] Run the deterministic source-checkout quickstart.
- [x] Run the deterministic lifecycle against the installed wheel from a clean directory.
- [x] Check the package index for the availability of `vrse-pytorch` during release preparation.

## Evidence

- [x] Freeze the Phase-3B protocol, result and SHA-256 manifest.
- [x] Keep Phase-3A invalid artifacts and the earlier spectral-norm failure traceable.
- [x] Publish human-readable per-seed metrics and three result figures.
- [ ] Obtain an independent third-party reproduction without overwriting the snapshot (post-release).

## Documentation

- [x] Add English and Chinese first-reader READMEs.
- [x] Explain the lifecycle before introducing implementation terminology.
- [x] Separate routing guarantees from statistical and domain-safety claims.
- [x] Document data provenance and the C-MAPSS redistribution restriction.
- [x] Add the canonical repository URL.
- [x] Add the version-specific Zenodo DOI after publication.

## Anonymous release audit

- [x] Remove personal names, emails, absolute paths and machine identifiers from tracked files.
- [x] Inspect image, archive and model/checkpoint byte content for identifying metadata.
- [ ] Optionally create a dedicated signing key for later releases under the stable project identity.
- [x] Create the annotated `v0.1.0a1` tag and attach the audited artifacts.
- [x] Submit the exact release to Software Heritage and publish it in Zenodo.
- [x] Record release artifact SHA-256 values in the Zenodo archive.

## Release boundary

Do not advertise production safety, universal OOD detection, general continual-learning
success, C-MAPSS SOTA, or robustness to malicious data. The accurate release claim is:

> VRSE is a research implementation of isolated shadow learning, independent promotion
> validation, regional deployment permission and exact frozen fallback, with a frozen
> five-seed demonstration on one 24-dimensional industrial simulation benchmark.
