# Changelog

All notable public-facing changes to VRSE will be documented here.

## [0.1.0a1] — 2026-07-29

First release candidate for independent review.

### Added

- `VRSEModel` lifecycle: fit, isolated observe, evaluate, atomic promote, route and revoke.
- Immutable promotion proposals bound to the evaluated baseline, configuration,
  candidate posterior and authorized region.
- Exact frozen-baseline routing outside authorized regions.
- One-dimensional observed-span and high-dimensional KNN feature regions.
- SNGP-style spectrally normalized features and an RFF Bayesian residual head.
- Deterministic quickstart and a frozen C-MAPSS FD002 reproduction path.
- Phase-3B five-seed evidence with stable adaptation and a reversed-label negative control.

### Validated scope

- Supervised scalar regression.
- One active regional residual expert.
- CPU reference implementation.
- One promotion event in a 24-dimensional industrial simulation case study.

### Known limitations

- No classification, delayed-label, adversarial-stream or multi-expert validation.
- No distribution-free deployment-risk certificate.
- No production safety or C-MAPSS state-of-the-art claim.
