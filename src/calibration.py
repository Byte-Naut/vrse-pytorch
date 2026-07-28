"""Distribution-free one-sided tolerance limits and exact binomial confidence
bounds (Stage-3B amendment, results/STAGE3_PROTOCOL.md).

`tolerance_limit_tau` fixes the finite-sample under-coverage of a plain
empirical percentile (Stage-3A's failure mode): a sample quantile is only a
point estimate of the population quantile, with no confidence margin, so a
plain 95th-percentile threshold clears a FRESH held-out draw's true 95%
target less than half the time. Wilks' method instead picks the order
statistic that, with 95% confidence over the calibration draw, upper-bounds
the true (1-p0) tail mass -- i.e. the resulting tau actually delivers >=p0
coverage on new draws with the stated confidence, rather than merely in
expectation.

`clopper_pearson_upper` / `clopper_pearson_lower` give the corresponding
one-sided exact confidence bounds used by the precondition audit, so a
borderline point estimate isn't treated as a failure unless the data gives
real evidence against the target.
"""

import torch
from scipy.stats import beta as beta_dist
from scipy.stats import binom


def tolerance_limit_tau(u_calib: torch.Tensor, p0: float = 0.95, confidence: float = 0.95) -> float:
    """Wilks' one-sided distribution-free tolerance limit.

    Given n i.i.d. calibration draws of u, returns tau = u_(k) (the k-th
    smallest, 1-indexed) such that, with `confidence` probability over the
    calibration draw, a fresh draw's true P(u <= tau) >= p0.

    k = min{ j : P[Binomial(n, p0) >= j] <= 1 - confidence }
      = binom.ppf(confidence, n, p0) + 1        (1-indexed order statistic)
    """
    n = u_calib.numel()
    delta = 1.0 - confidence
    k = int(binom.ppf(confidence, n, p0)) + 1
    if k > n:
        raise ValueError(
            f"Tolerance limit requires the {k}-th of {n} order statistics, which exceeds the "
            f"calibration sample size -- increase the calibration set size."
        )
    sorted_u, _ = torch.sort(u_calib.reshape(-1))
    return sorted_u[k - 1].item()


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact upper confidence bound on a true binomial proportion."""
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(confidence, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact lower confidence bound on a true binomial proportion."""
    if k <= 0:
        return 0.0
    return float(beta_dist.ppf(1.0 - confidence, k, n - k + 1))
